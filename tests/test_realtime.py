import asyncio
import base64
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Interface, IPv4Network
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave
from io import BytesIO

import numpy as np
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from jr_realtime_client import stream_speech
from provider.audio import local_ffmpeg
from provider.realtime import create_app, directed_chunks, voice_options
from provider.server import api_boundary


class RealtimeTests(unittest.TestCase):
    def serve(self, events, speechcpp=True, *, text="你好。Hello.", public=False):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                model = {"id": "drama-3"}
                if public:
                    model["owned_by"] = "jr-drama3"
                if speechcpp:
                    model["speech"] = {}
                self.wfile.write(json.dumps({"data": [model]}).encode())

            def do_POST(self):
                calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return stream_speech(text, "zh", 42, port=server.server_port), calls

    def test_complete_stream_and_seed_on_both_engines(self):
        pcm = bytes(range(20))
        events = [
            {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()},
            {"type": "speech.audio.done", "samples": 10, "stop": "complete"},
        ]
        for speechcpp in (True, False):
            stream, calls = self.serve(events, speechcpp)
            with closing(stream):
                self.assertEqual(list(stream), [pcm])
            self.assertEqual((calls[0] if speechcpp else calls[0]["options"])["seed"], 42)

    def test_rejects_eof_truncation_and_malformed_pcm(self):
        delta = {"type": "speech.audio.delta", "audio": "AAAAAA=="}
        for events in (
            [delta],
            [delta, {"type": "speech.audio.done", "stop": "max_seconds"}],
            [delta, {"type": "speech.audio.done", "samples": 5}],
            [{"type": "speech.audio.delta", "audio": "AA=="}],
        ):
            stream, _ = self.serve(events)
            with closing(stream), self.assertRaises(RuntimeError):
                list(stream)

    def test_client_sends_empty_text_only_to_the_public_gateway(self):
        events = [{"type": "speech.audio.delta", "audio": "AAA="},
                  {"type": "speech.audio.done", "samples": 1, "stop": "complete"}]
        stream, calls = self.serve(events, text="", public=True)
        with closing(stream):
            self.assertEqual(list(stream), [bytes(2)])
        self.assertEqual(calls[0]["input"], "")
        stream, calls = self.serve(events, text="")
        with closing(stream), self.assertRaises(ValueError):
            list(stream)
        self.assertEqual(calls, [])


class SentenceGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_hermes_file_formats_return_decodable_audio(self):
        tone = (12000 * np.sin(2 * np.pi * 440 * np.arange(11025) / 44100)).astype("<i2")

        async def generate(request):
            delta = {"type": "speech.audio.delta", "audio": base64.b64encode(tone.tobytes()).decode()}
            done = {"type": "speech.audio.done", "stop": "complete", "samples": len(tone)}
            return web.Response(text=f"data: {json.dumps(delta)}\n\ndata: {json.dumps(done)}\n\n",
                                content_type="text/event-stream")

        native = web.Application()
        native.router.add_post("/v1/audio/speech", generate)
        async with TestServer(native) as engine, TestClient(TestServer(create_app("5lang", engine.port))) as client:
            for fmt, mime, rate in (("mp3", "audio/mpeg", 44100), ("opus", "audio/ogg", 48000),
                                    ("aac", "audio/aac", 44100), ("flac", "audio/flac", 44100)):
                with self.subTest(fmt=fmt):
                    resp = await client.post("/v1/audio/speech", json={
                        "model": "drama-3", "voice": "zh", "input": "你好。", "response_format": fmt})
                    self.assertEqual((resp.status, resp.content_type), (200, mime))
                    self.assertEqual(int(resp.headers["X-Sample-Rate"]), rate)
                    decoded = await asyncio.to_thread(subprocess.run, [
                        local_ffmpeg(), "-v", "error", "-i", "pipe:0", "-ar", "44100", "-f", "s16le", "pipe:1"
                    ], input=await resp.read(), capture_output=True, check=True)
                    pcm = np.frombuffer(decoded.stdout, dtype="<i2")
                    self.assertGreaterEqual(len(pcm), len(tone) - 1)
                    self.assertGreater(int(pcm.max()), 5000)
            for fmt, stream, param in (("unknown", "audio", "response_format"), ([], "audio", "response_format"),
                                       ("mp3", "sse", "stream_format")):
                resp = await client.post("/v1/audio/speech", json={"input": "Hello.",
                                         "response_format": fmt, "stream_format": stream})
                self.assertEqual(resp.status, 400)
                self.assertEqual((await resp.json())["error"]["param"], param)

    async def test_overlong_reference_returns_actionable_client_error(self):
        async def reject(request):
            event = {"type": "error", "error": {"message": "Drama 3 required prompt leaves no room to generate"}}
            return web.Response(text="data: " + json.dumps(event) + "\n\n", content_type="text/event-stream")

        native = web.Application()
        native.router.add_post("/v1/audio/speech", reject)
        async with TestServer(native) as server:
            async with TestClient(TestServer(create_app("5lang", engine_port=server.port))) as client:
                reply = await client.post("/v1/audio/speech", json={"input": "Hello."})
                self.assertEqual(reply.status, 400)
                self.assertIn("matching transcript", (await reply.json())["error"]["message"])

    async def test_switch_is_exclusive_and_failed_switch_keeps_the_active_model(self):
        entered = threading.Event()
        release = threading.Event()
        selected = []

        def switch(target):
            selected.append(target)
            if target == "nvfp4":
                raise FileNotFoundError("uninstalled model")
            entered.set()
            if not release.wait(5):
                raise RuntimeError("timed out")

        async with TestClient(TestServer(create_app("q4", switch=switch))) as client:
            pending = asyncio.create_task(client.post("/v1/models/select", json={"model": "drama-3-5lang"}))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                resp = await client.post("/v1/audio/speech", json={"input": "Hello."})
                self.assertEqual(resp.status, 429)
                resp = await client.post("/v1/models/select", json={"model": "drama-3-q4"})
                self.assertEqual(resp.status, 429)
            finally:
                release.set()
                resp = await pending
            self.assertEqual((await resp.json())["backend"], "5lang")
            resp = await client.post("/v1/models/select", json={"model": "drama-3-nvfp4"})
            self.assertEqual(resp.status, 503)
            resp = await client.get("/v1/models")
            data = (await resp.json())["data"]
            self.assertEqual([x["id"] for x in data if x["active"]], ["drama-3", "drama-3-5lang"])
            self.assertEqual(selected, ["5lang", "nvfp4"])

    def test_decoration_chunks_are_removed_before_directions_and_seeding(self):
        self.assertEqual(directed_chunks("好的。😊\n---\n👨‍👩‍👧‍👦\n再见。～", "whisper"),
                         ["[whisper] 好的。", "[whisper] 再见。"])
        self.assertEqual(directed_chunks("好的。\u200d\u200d\u200d", ""), ["好的。"])
        self.assertEqual(directed_chunks("<|speaker:1|> <whisper>😊。</whisper>", ""), [])
        self.assertEqual(directed_chunks("Good😊morning.", ""), ["Good morning."])
        for text in ("42", "好", "I", "<whisper>好。</whisper>", "<|phoneme_start|>ni3<|phoneme_end|>",
                     "[laughs] Yes."):
            self.assertEqual(directed_chunks(text, ""), [text])
        for text in ("", " \n\t", "<|speaker:2|>", "<whisper> </whisper>", "[laughs]",
                     "<|speaker:2|> <whisper>[icy calm] … 😊</whisper>",
                     "\u3164\u200b", "[]", "<|phoneme_start|><|phoneme_end|>",
                     "<|phoneme_start|>  <|phoneme_end|>"):
            with self.subTest(text=text):
                self.assertEqual(directed_chunks(text, "whisper"), [])
        with self.assertRaises(ValueError):
            directed_chunks("<whisper>", "")

    async def test_decorations_alone_return_valid_silence_without_an_engine(self):
        async with TestClient(TestServer(create_app("nvfp4", engine_port=1))) as client:
            for text, fmt, stream, accept in (("", "pcm", "audio", "application/octet-stream"),
                                              (" \t\n", "pcm", "audio", "*/*"),
                                              ("[whisper]", "pcm", "sse", "*/*"),
                                              ("<whisper></whisper>", "wav", "audio", "*/*"),
                                              ("😊\n。。。\n🏳️‍🌈\n---", "wav", "audio", "*/*")):
                response = await client.post("/v1/audio/speech", headers={"Accept": accept},
                    json={"input": text, "instructions": "whisper",
                          "response_format": fmt, "stream_format": stream})
                self.assertEqual(response.status, 200)
                raw = await response.read()
                rate = int(response.headers["X-Sample-Rate"])
                if fmt == "wav":
                    with wave.open(BytesIO(raw)) as w:
                        self.assertEqual(w.getframerate(), rate)
                        raw = w.readframes(w.getnframes())
                elif stream == "sse":
                    delta, done = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith(b"data:")]
                    raw = base64.b64decode(delta["audio"])
                    self.assertEqual(done["samples"], len(raw) // 2)
                self.assertEqual(raw, bytes(2 * (rate // 50)))
            for body in ({}, {"input": None}, {"input": []}, {"input": " " * 4097}, {"input": "[whisper"}):
                self.assertEqual((await client.post("/v1/audio/speech", json=body)).status, 400)

    async def test_sdk_pcm_stream_preserves_pitch_duration_and_final_samples(self):
        release_tail = asyncio.Event()
        tone = np.rint(12000 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype("<i2")

        async def generate(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for i, samples in enumerate(np.array_split(tone, 7)):
                if i == 1:
                    await asyncio.wait_for(release_tail.wait(), 5)
                event = {"type": "speech.audio.delta", "audio": base64.b64encode(samples.tobytes()).decode()}
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b'data: {"type":"speech.audio.done","samples":44100,"stop":"complete"}\n\n')
            return response

        native = web.Application()
        native.router.add_post("/v1/audio/speech", generate)
        async with TestServer(native) as engine, TestClient(TestServer(create_app("nvfp4", engine.port))) as client:
            async with client.post("/v1/audio/speech", json={"input": "Hi.", "response_format": "pcm"},
                                   headers={"Accept": "application/octet-stream"}) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers["X-Sample-Rate"], "24000")
                self.assertEqual(resp.content_type, "application/octet-stream")
                try:
                    first = await asyncio.wait_for(resp.content.readexactly(128), 3)
                    blank = await client.post("/v1/audio/speech", json={"input": "[whisper]"})
                    self.assertEqual(blank.status, 200)
                    self.assertEqual(await blank.read(), bytes(2 * (44100 // 50)))
                    busy = await client.post("/v1/audio/speech", json={"input": "Still busy?"})
                    self.assertEqual(busy.status, 429)
                finally:
                    release_tail.set()
                samples = np.frombuffer(first + await resp.read(), dtype="<i2")
        self.assertEqual(len(samples), 24000)
        expected = 12000 * np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)
        self.assertLess(np.max(np.abs(expected[128:-128] - samples[128:-128])), 8)

    async def test_lan_access_requires_configured_peer_host_and_origin(self):
        lan = IPv4Interface("192.168.1.200/24")
        async def ok(request):
            return web.Response()
        for interface, peer, host, origin, status in (
            (lan, "192.168.1.12", "192.168.1.200", "", 200),
            (lan, "192.168.100.12", "192.168.1.200", "", 200),
            (lan, "127.0.0.1", "localhost", "", 200),
            (lan, "192.168.10.12", "192.168.1.200", "", 403),
            (lan, "192.168.1.12", "elsewhere.test", "", 403),
            (lan, "192.168.1.12", "192.168.1.200", "http://elsewhere.test", 403),
            (None, "192.168.100.215", "192.168.1.241", "", 200),
            (None, "192.168.100.215", "speech.example.org", "https://speech.example.org", 200),
            (None, "192.168.100.215", "speech.example.org", "https://unrelated.example.org", 403),
        ):
            app = create_app("nvfp4", lan=interface, extra_networks=(IPv4Network("192.168.100.0/24"),))
            request = make_mocked_request("GET", "/health", {"Host": host, "Origin": origin}, app=app)
            self.assertEqual((await api_boundary(request.clone(remote=peer), ok)).status, status)

    async def test_persistent_voice_versions_match_comfy_and_round_trip_stereo(self):
        buf = BytesIO()
        with wave.open(buf, "wb") as w:
            w.setparams((2, 2, 16000, 0, "NONE", ""))
            w.writeframes(struct.pack("<6h", -32768, 16384, 0, -8192, 32767, 0))
        pair = {"voice_ref": {"type": "base64", "data": base64.b64encode(buf.getvalue()).decode()},
                "reference_text": "Morning."}
        payload = dict(pair, name="Narrator", language="en")
        # Version produced by the original ComfyUI database format for these exact samples and metadata.
        expected = "49827f50f3992dd70cc7b65b50164227c55e627301d23d5d89ec6a7c26f53b74"
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "voices.sqlite3"
            async with TestClient(TestServer(create_app("nvfp4", voice_db=db))) as client:
                for _ in range(2):
                    resp = await client.post("/v1/audio/voices", json=payload)
                    self.assertEqual((await resp.json())["id"], expected)
                resp = await client.post("/v1/audio/voices", json=dict(payload, reference_text="Hi."))
                self.assertNotEqual((await resp.json())["id"], expected)
            async with TestClient(TestServer(create_app("nvfp4", voice_db=db))) as client:
                resp = await client.get("/v1/audio/voices")
                rows = (await resp.json())["data"]
                self.assertEqual([v["id"] for v in rows].count(expected), 1)
                self.assertEqual(len(rows), 4)
                for fields in ({"voice": "a"*64}, {"voice": expected, **pair}, {"voice": "Narrator"}):
                    resp = await client.post("/v1/audio/speech", json=dict(input="Hello.", **fields))
                    self.assertEqual(resp.status, 400)
                self.assertEqual(voice_options({"voice": expected}, "nvfp4", db), pair)
            async with TestClient(TestServer(create_app("q4", voice_db=db))) as client:
                resp = await client.post("/v1/audio/voices", json=payload)
                self.assertEqual((await resp.json())["id"], expected)
                resp = await client.get("/v1/audio/voices")
                self.assertEqual((await resp.json())["data"], rows)

    def test_direction_follows_each_speaker_and_chunk(self):
        text = "<|speaker:1|> 好的。稍等。<|speaker:0|> <whisper>On my way.</whisper>"
        self.assertEqual(directed_chunks(text, " icy calm "), [
            "<|speaker:1|> [icy calm] 好的。", "<|speaker:1|> [icy calm] 稍等。",
            "<|speaker:0|> [icy calm] <whisper>On my way.</whisper>",
        ])
        for bad in ["[whisper]", "a" * 201, 2]:
            with self.assertRaises(ValueError):
                directed_chunks("Hi.", bad)

    async def test_custom_reference_precedence_and_invalid_pairs(self):
        buf = BytesIO()
        with wave.open(buf, "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", ""))
            w.writeframes(bytes(400))
        ref = {"type": "base64", "data": base64.b64encode(buf.getvalue()).decode()}
        pair = {"voice_ref": ref, "reference_text": "Good morning."}
        self.assertEqual(voice_options(dict(voice="zh", **pair), "nvfp4"), pair)
        for backend in ("q4", "nvfp4"):
            async with TestClient(TestServer(create_app(backend))) as client:
                for invalid in ({"voice_ref": ref}, {"reference_text": "orphan"},
                                dict(pair, reference_text="bad\x00text"),
                                dict(pair, voice_ref={"type": "path", "path": "ignored.wav"}),
                                dict(pair, voice_ref={"type": "base64", "data": "%%%"}),
                                dict(pair, voice_ref={"type": "base64", "data": base64.b64encode(buf.getvalue()[:-20]).decode()})):
                    resp = await client.post("/v1/audio/speech", json=dict(invalid, input="Hi."))
                    self.assertEqual(resp.status, 400, await resp.text())

    async def test_q4_inline_and_saved_references_use_the_resident_engine(self):
        loaded = {}
        requests = []
        fail_upload = True

        async def models(request):
            return web.json_response({"data": [{"id": "drama-3", "speech": {"voices": [
                {"name": name} for name in loaded]}}]})

        async def add(request):
            form = await request.post()
            if fail_upload:
                return web.Response(status=500, text="registration failed")
            loaded[form["name"]] = form["file"].file.read()
            return web.json_response({"held": {}})

        async def synthesize(request):
            body = await request.json()
            self.assertIn(body["voice"], loaded)
            self.assertNotIn("voice_ref", body)
            self.assertNotIn("reference_text", body)
            requests.append(body)
            return web.Response(text='data: {"type":"speech.audio.delta","audio":"AQA="}\n\n'
                                     'data: {"type":"speech.audio.done","samples":1,"stop":"complete"}\n\n')

        native = web.Application()
        native.router.add_get("/v1/models", models)
        native.router.add_post("/speech/voices", add)
        native.router.add_post("/v1/audio/speech", synthesize)
        buf = BytesIO()
        with wave.open(buf, "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", ""))
            w.writeframes(bytes(400))
        # Extra RIFF metadata in an uploaded WAV is absent from the copy reconstructed from SQLite.
        raw = buf.getvalue()
        tagged = raw[:4] + struct.pack("<I", len(raw) + 4) + raw[8:12] + b"JUNK\x04\x00\x00\x00test" + raw[12:]
        ref = {"type": "base64", "data": base64.b64encode(tagged).decode()}
        pair = {"voice_ref": ref, "reference_text": "Morning."}
        with tempfile.TemporaryDirectory() as tmp, patch("provider.q4_voice.encode_reference",
                                                        return_value=b"GGUF fixture") as encode:
            db = Path(tmp) / "voices.sqlite3"
            async with TestServer(native) as engine, TestClient(TestServer(create_app("q4", engine.port, voice_db=db))) as client:
                async def speak(fields):
                    async with client.post("/v1/audio/speech", json=dict(
                            input="你好。Hello.", instructions="whisper", seed=19, **fields)) as response:
                        return response.status, await response.read()

                self.assertEqual((await speak(pair))[0], 502)
                self.assertFalse(requests)
                fail_upload = False
                complete = (200, b"\x01\x00" * 2)
                self.assertEqual(await speak(pair), complete)
                encode.assert_awaited_with(buf.getvalue(), "Morning.")
                first = requests[0]["voice"]
                self.assertEqual(loaded, {first: b"GGUF fixture"})
                self.assertEqual([r["input"] for r in requests], ["[whisper] 你好。", "[whisper] Hello."])
                self.assertEqual([r["seed"] for r in requests], [19, 20])
                saved = await client.post("/v1/audio/voices", json=dict(name="Saved", **pair))
                voice_id = (await saved.json())["id"]
                self.assertEqual(await speak({"voice": voice_id}), complete)
                self.assertEqual(requests[-1]["voice"], first)
                self.assertEqual(encode.await_count, 2)  # Failed upload, then successful retry.
                self.assertEqual(await speak(dict(pair, reference_text="Different.")), complete)
                self.assertNotEqual(requests[-1]["voice"], first)
                loaded.clear()  # An engine restart drops its native voices.
                self.assertEqual(await speak({"voice": voice_id}), complete)
                self.assertEqual(list(loaded), [first])

    async def test_audio_and_wav_preserve_both_sentences_and_seed_rollover(self):
        requests = []
        pcm = b"\x01\x00" * 100

        async def generate(request):
            requests.append(await request.json())
            delta = {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()}
            done = {"type": "speech.audio.done", "samples": 100, "stop": "complete"}
            return web.Response(
                content_type="text/event-stream",
                text=f"data: {json.dumps(delta)}\n\ndata: {json.dumps(done)}\n\n",
            )

        native = web.Application()
        native.router.add_post("/v1/audio/speech", generate)
        async with TestServer(native) as server:
            async with TestClient(TestServer(create_app("q4", server.port))) as client:
                for fmt in ("pcm", "wav"):
                    requests.clear()
                    async with client.post(
                        "/v1/audio/speech",
                        json={"input": "现在出发。\n[whisper] …\n\n<whisper></whisper>\nSee you soon.",
                              "response_format": fmt, "seed": 2**32 - 1},
                    ) as resp:
                        self.assertEqual(resp.status, 200)
                        raw = await resp.read()
                        if fmt == "wav":
                            with wave.open(BytesIO(raw)) as audio:
                                self.assertEqual(audio.getframerate(), 44100)
                                raw = audio.readframes(audio.getnframes())
                        self.assertEqual(raw, pcm * 2)
                    self.assertEqual([r["input"] for r in requests], ["现在出发。", "See you soon."])
                    self.assertEqual([r["seed"] for r in requests], [2**32 - 1, 0])
            async with TestClient(TestServer(create_app("nvfp4", server.port))) as client:
                # The native default split at 128 codepoints could isolate the tail.
                line = "Please listen " * 10 + "[whisper]"
                requests.clear()
                resp = await client.post("/v1/audio/speech", json={"input": line, "response_format": "wav"})
                self.assertEqual(resp.status, 200)
                self.assertEqual([r["input"] for r in requests], [line])
                self.assertGreaterEqual(requests[0]["options"]["text_chunk_size"], len(line))
