import asyncio
from contextlib import closing
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
import wave

import numpy as np
import soundfile as sf
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from provider.audio import convert
from provider.backend import loopback_url
from provider.server import create_app
from provider.speech import ApiError, VOICES, build_graph, speech_params


def sample_flac():
    buf = BytesIO()
    t = np.arange(44100) / 44100
    sf.write(buf, .18 * np.sin(2 * np.pi * 330 * t), 44100, format="FLAC")
    return buf.getvalue()


class ComfyStub:
    def __init__(self):
        self.app = web.Application()
        self.jobs, self.cancelled = {}, []
        self.ready, self.pending, self.reject = True, False, False
        self.submit_delay = 0.
        self.reason, self.seen, self.stopped = "eos", asyncio.Event(), asyncio.Event()
        self.audio = sample_flac()
        self.app.router.add_get("/object_info/JRDrama3Generate", self.info)
        self.app.router.add_post("/prompt", self.post)
        self.app.router.add_get("/history/{job}", self.history)
        self.app.router.add_post("/api/jobs/{job}/cancel", self.cancel)
        self.app.router.add_get("/view", self.view)

    async def info(self, request):
        return web.json_response({"JRDrama3Generate": {}} if self.ready else {})

    async def post(self, request):
        body = await request.json()
        self.seen.set()
        if self.submit_delay:
            await asyncio.sleep(self.submit_delay)
        self.jobs[body["prompt_id"]] = body["prompt"]
        if self.reject:
            raise web.HTTPBadRequest()
        return web.json_response({"prompt_id": body["prompt_id"]})

    async def history(self, request):
        job = request.match_info["job"]
        if self.pending:
            return web.json_response({})
        graph = self.jobs[job]
        outputs = {key: {"text": [json.dumps({"finish_reason": self.reason, "seconds": 1.})]}
                   for key in graph if key.startswith("report_")}
        if "find" in graph:
            outputs["voice_id"] = {"text": [graph["find"]["inputs"]["query"]]}
        outputs["audio"] = {"audio": [{"filename": "example.flac", "type": "temp", "subfolder": ""}]}
        return web.json_response({job: {"status": {"status_str": "success"}, "outputs": outputs}})

    async def cancel(self, request):
        self.cancelled.append(request.match_info["job"])
        self.stopped.set()
        return web.json_response({"cancelled": True})

    async def view(self, request):
        assert dict(request.query) == {"filename": "example.flac", "type": "temp", "subfolder": ""}
        return web.Response(body=self.audio, content_type="audio/flac")


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "voices.sqlite3"
        self.voice = "e" * 64
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute("CREATE TABLE voices (row_id INTEGER,voice_id TEXT,name TEXT,language TEXT,description TEXT,tags TEXT)")
            con.execute("INSERT INTO voices VALUES (1,?,?,?,?,?)", (self.voice, "Saved voice", "zh", "calm", "demo"))
        self.backends, self.urls = [], []
        for _ in range(2):
            stub = ComfyStub()
            server = TestServer(stub.app)
            await server.start_server()
            self.addAsyncCleanup(server.close)
            self.backends.append(stub)
            self.urls.append(str(server.make_url("/")).rstrip("/"))
        self.body = dict(model="drama-3", input="Hello world.", voice="alloy", response_format="wav")

    async def client(self, timeout=5):
        app = create_app(primary=self.urls[0], fallback=self.urls[1], voice_db=self.db, timeout=timeout, poll=.01)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def test_list_catalog_and_validate_input(self):
        client = await self.client()
        voices = await (await client.get("/v1/audio/voices")).json()
        self.assertEqual({v["id"] for v in voices["data"]}, set(VOICES) | {self.voice})
        models = await (await client.get("/v1/models")).json()
        self.assertEqual([m["id"] for m in models["data"]], ["drama-3"])
        health = await (await client.get("/health")).json()
        self.assertEqual(health["selected_backend"], self.urls[0])
        for name, value in [("speed", .2), ("voice", "f" * 64), ("response_format", "exe"),
                            ("seed", True), ("input", ""), ("stream_format", "sse")]:
            response = await client.post("/v1/audio/speech", json={**self.body, name: value})
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())["error"]["param"], name)
        self.assertEqual((await client.get("/v1/models", headers={"Host": "example.org"})).status, 403)
        self.assertEqual((await client.get("/v1/models", headers={"Origin": "http://example.org"})).status, 403)
        self.assertEqual(self.backends[0].jobs, {})

    @unittest.skipUnless(shutil.which("ffmpeg"), "Requires local FFmpeg")
    async def test_fallback_unload_saved_voice_and_primary_preference(self):
        client = await self.client()
        self.backends[0].ready = False
        response = await client.post("/v1/audio/speech", json={**self.body, "voice": {"id": self.voice},
                                      "instructions": "Gently.", "seed": 52, "speed": 1.25})
        self.assertEqual(response.status, 200, await response.text() if response.status != 200 else "")
        self.assertEqual(response.headers["X-Drama3-Backend"], self.urls[1])
        with wave.open(BytesIO(await response.read())) as audio:
            self.assertEqual((audio.getframerate(), audio.getnchannels()), (44100, 1))
            self.assertTrue(34000 < audio.getnframes() < 37500)
        graph = self.backends[1].jobs[response.headers["X-Request-ID"]]
        self.assertEqual(graph["audio"]["inputs"]["audio"], ["unload", 1])
        self.assertEqual(graph["unload"]["inputs"]["after_audio"], ["take_0", 0])
        self.assertEqual(graph["take_0"]["inputs"]["direction"], "Gently.")
        self.assertEqual(graph["take_0"]["inputs"]["cast"], ["cast", 0])
        self.backends[0].ready = True
        response = await client.post("/v1/audio/speech", json=self.body)
        self.assertEqual(response.status, 200)
        graph = self.backends[0].jobs[response.headers["X-Request-ID"]]
        self.assertNotIn("unload", graph)
        self.assertEqual(graph["take_0"]["inputs"]["mode"], "design")

    async def test_no_resubmit_or_partial_success(self):
        client = await self.client()
        self.backends[0].reason = "length"
        response = await client.post("/v1/audio/speech", json=self.body)
        self.assertEqual(response.status, 502)
        self.assertEqual((await response.json())["error"]["code"], "incomplete_speech")
        self.backends[0].reject = True
        response = await client.post("/v1/audio/speech", json=self.body)
        self.assertEqual(response.status, 502)
        self.assertFalse(self.backends[1].jobs)
        self.assertEqual(len(self.backends[0].jobs), 2)

    async def test_timeout_rate_limit_and_scoped_cancel(self):
        client = await self.client(timeout=.35)
        stub = self.backends[0]
        stub.pending = True
        first = asyncio.create_task(client.post("/v1/audio/speech", json=self.body))
        await asyncio.wait_for(stub.seen.wait(), 1.)
        busy = await client.post("/v1/audio/speech", json=self.body)
        self.assertEqual(busy.status, 429)
        self.assertEqual(busy.headers["Retry-After"], "5")
        timeout = await first
        self.assertEqual(timeout.status, 504)
        self.assertEqual(stub.cancelled, list(stub.jobs))
        self.assertFalse(self.backends[1].jobs)

    async def test_disconnect_during_submission_cancels_after_acknowledgement(self):
        client = await self.client()
        stub = self.backends[0]
        stub.pending, stub.submit_delay = True, .12
        speech = asyncio.create_task(client.post("/v1/audio/speech", json=self.body))
        await asyncio.wait_for(stub.seen.wait(), 1.)
        speech.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await speech
        await asyncio.wait_for(stub.stopped.wait(), 2.)
        self.assertEqual(stub.cancelled, list(stub.jobs))
        self.assertEqual(len(stub.jobs), 1)

    def test_long_input_is_one_comfy_job_with_linked_takes(self):
        req = speech_params({**self.body, "input": "我们再来一遍。 " * 90, "instructions": "warm", "seed": 17})
        graph, reports = build_graph(req, unload=True)
        self.assertGreater(len(reports), 2)
        self.assertIn("warm", graph["take_0"]["inputs"]["direction"])
        self.assertEqual(graph["take_1"]["inputs"]["previous"], ["take_0", 1])
        self.assertEqual(graph["take_1"]["inputs"]["mode"], "continue")
        self.assertEqual(graph["take_1"]["inputs"]["seed"], 18)
        self.assertEqual(graph["unload"]["inputs"]["after_audio"], [f"join_{len(reports)-1}", 0])
        for url in ("https://127.0.0.1:8", "http://example.test", "http://localhost:8/path", "http://10.0.0.8"):
            with self.assertRaises(ValueError):
                loopback_url(url)
        self.assertEqual(loopback_url("http://localhost:8188/"), "http://127.0.0.1:8188")

    @unittest.skipUnless(shutil.which("ffmpeg"), "Requires local FFmpeg")
    async def test_all_audio_encodings_and_extreme_speeds(self):
        source = sample_flac()
        for fmt in ("flac", "wav", "mp3", "aac", "opus", "pcm"):
            data = await convert(source, fmt, 1., "ffmpeg")
            if fmt == "pcm":
                self.assertEqual(len(data), 24000 * 2)
            elif fmt == "aac":
                self.assertEqual(data[0], 255)
            else:
                samples, rate = sf.read(BytesIO(data))
                self.assertTrue(.97 < len(samples) / rate < 1.04)
        for speed in (.25, 4.):
            data = await convert(source, "pcm", speed, "ffmpeg")
            self.assertLess(abs(len(data) / 48000 - 1. / speed), .15)
        with self.assertRaises(ApiError):
            await convert(b"<html/>", "wav", 1., "ffmpeg")
