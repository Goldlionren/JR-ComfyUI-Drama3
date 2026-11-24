import base64
import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
import wave

import numpy as np
import soxr
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from jr_benchmark_cases import FIXTURES, REFERENCES
from jr_realtime_client import RATE
from provider.realtime import create_app, directed_chunks
from provider.reference_echo import ReferenceEcho


class ReferenceEchoTests(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        name, _, transcript, _ = REFERENCES["en"]
        path = FIXTURES / name
        cls.voice = {"voice_ref": {"type": "base64", "data": base64.b64encode(path.read_bytes()).decode()},
                     "reference_text": transcript}
        with wave.open(str(path)) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            clip = soxr.resample(samples.astype(np.float32), w.getframerate(), RATE)
        cls.echo = (clip * .65).astype("<i2").tobytes()

    def test_partial_copy_inside_dialogue_and_intentional_quotation(self):
        guard = ReferenceEcho.from_voice(self.voice)
        cut = self.echo[2 * RATE:8 * RATE]
        self.assertTrue(guard.matches(bytes(2 * RATE * 7) + cut + bytes(2 * RATE), RATE))
        self.assertFalse(guard.matches(bytes(len(self.echo)), RATE))
        self.assertFalse(guard.matches(self.echo[:RATE], RATE))
        self.assertFalse(guard.applies(self.voice["reference_text"].upper()))
        self.assertTrue(guard.applies("I opened another window."))

    def test_markdown_list_and_emphasis_keep_dialogue_together(self):
        self.assertEqual(directed_chunks("1. **收到**。\n2) 明天见。", ""), ["收到。", "明天见。"])
        for line in ("42", "3.14", "2 * 3", "在 1.5 米处。", "[quietly] Yes."):
            self.assertEqual(directed_chunks(line, ""), [line])

    async def test_retries_only_the_matching_sentence_before_delivery(self):
        clean = b"\x08\x00" * 100
        for backend, fmt in (("5lang", "sse"), ("nvfp4", "audio"), ("q4", "sse")):
            with self.subTest(backend=backend):
                calls = []

                async def native_speech(request):
                    body = await request.json()
                    opts = body if backend == "q4" else body["options"]
                    calls.append((body["input"], opts["seed"]))
                    pcm = self.echo if len(calls) == 2 else clean
                    events = [{"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()},
                              {"type": "speech.audio.done", "samples": len(pcm) // 2}]
                    return web.Response(text="".join(f"data: {json.dumps(e)}\n\n" for e in events))

                native = web.Application()
                native.router.add_post("/v1/audio/speech", native_speech)
                with patch("provider.realtime.load_q4_voice", new=AsyncMock(return_value="fixture")):
                    async with TestServer(native) as engine, TestClient(TestServer(create_app(backend, engine.port))) as client:
                        response = await client.post("/v1/audio/speech", json={
                            **self.voice, "input": "First. Second. Third.", "stream_format": fmt, "seed": 2**32-1})
                        self.assertEqual(response.status, 200)
                        raw = await response.read()
                if fmt == "sse":
                    events = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith(b"data:")]
                    self.assertEqual(events[-1]["samples"], 300)
                    raw = b"".join(base64.b64decode(e["audio"]) for e in events[:-1])
                self.assertEqual(raw, clean * 3)
                self.assertEqual([c[0] for c in calls], ["First.", "Second.", "Second.", "Third."])
                self.assertEqual([calls[i][1] for i in (0, 1, 3)], [2**32-1, 0, 1])
                self.assertNotEqual(calls[1][1], calls[2][1])

    async def test_exhaustion_rejects_a_wav_and_ends_a_started_stream(self):
        calls = []

        async def native_speech(request):
            body = await request.json()
            calls.append(body["options"]["seed"])
            data = base64.b64encode(b"\x10\x00" if body["input"] == "First." else self.echo).decode()
            return web.Response(text=f'data: {{"type":"speech.audio.delta","audio":"{data}"}}\n\n'
                                     'data: {"type":"speech.audio.done"}\n\n')

        native = web.Application()
        native.router.add_post("/v1/audio/speech", native_speech)
        async with TestServer(native) as engine, TestClient(TestServer(create_app("5lang", engine.port))) as client:
            response = await client.post("/v1/audio/speech", json={
                **self.voice, "input": "Tomorrow.", "response_format": "wav"})
            self.assertEqual(response.status, 502)
            self.assertEqual((await response.json())["error"]["code"], "incomplete_speech")
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(set(calls)), 3)
            streamed = await client.post("/v1/audio/speech", json={
                **self.voice, "input": "First. Tomorrow.", "stream_format": "sse"})
            self.assertEqual(streamed.status, 200)
            events = [json.loads(line[5:]) for line in (await streamed.read()).splitlines() if line.startswith(b"data:")]
            self.assertEqual([e["type"] for e in events], ["speech.audio.delta", "error"])
            self.assertEqual(base64.b64decode(events[0]["audio"]), b"\x10\x00")
            self.assertEqual(events[-1]["code"], "incomplete_speech")
