"""Hermes client contract, using its real registry, configuration and OpenAI SDK."""

import array
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from speak import PROFILE_CONFIG, speak


@unittest.skipUnless(importlib.util.find_spec("hermes_cli"), "Run with the Hermes Python environment")
class HermesClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="drama3-hermes-test-")
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        shutil.copyfile(PROFILE_CONFIG, self.home / "config.yaml")
        self.enterContext(patch.dict(os.environ, {"HERMES_HOME": str(self.home), "NO_PROXY": "*"}))
        for name in ("HERMES_CONFIG", "HERMES_CONFIG_PATH", "HERMES_PROFILE", "HERMES_SESSION_PLATFORM"):
            os.environ.pop(name, None)

    def intercept(self, body):
        import httpx

        self.requests = []

        def respond(transport, request):
            self.assertEqual(str(request.url), "http://192.168.1.241:8193/v1/audio/speech")
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["authorization"], "Bearer drama3-local-placeholder")
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"Content-Type": "application/octet-stream"})

        self.enterContext(patch.object(httpx.HTTPTransport, "handle_request", respond))

    def test_file_tool_forwards_script_model_direction_and_format(self):
        payload = io.BytesIO()
        with wave.open(payload, "wb") as wav:
            wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 1024)
        for suffix, expected in (("wav", "wav"), ("mp3", "mp3"), ("ogg", "opus")):
            with self.subTest(suffix=suffix):
                audio = payload.getvalue()
                if suffix != "wav":
                    codec = "libopus" if suffix == "ogg" else "libmp3lame"
                    audio = subprocess.run(
                        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-c:a", codec,
                         "-f", suffix, "pipe:1"], input=audio, capture_output=True, check=True,
                    ).stdout
                self.intercept(audio)
                target = self.home / ("spoken." + suffix)
                result = speak("Hello 你好", target, instructions="Speak warmly.")
                self.assertEqual(result["route"], "Hermes registry text_to_speech")
                self.assertEqual(self.requests[-1], {
                    "input": "Hello 你好", "model": "drama-3", "voice": "default",
                    "response_format": expected, "instructions": "Speak warmly.",
                })
                data = target.read_bytes()
                if suffix == "ogg":
                    self.assertTrue(data.startswith(b"OggS"))
                    self.assertIn(b"OpusHead", data)
                else:
                    self.assertEqual(data, audio)

    def test_hermes_pcm_stream_uses_24_khz(self):
        payload = array.array("h", [0, 10, -10, 5]).tobytes()
        self.intercept(payload)
        target = self.home / "spoken.pcm"
        result = speak("Hello.", target)
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(self.requests, [{"input": "Hello.", "model": "drama-3",
                                         "voice": "default", "response_format": "pcm"}])

    def test_wrong_model_is_refused_before_network(self):
        self.intercept(b"")
        config = self.home / "config.yaml"
        config.write_text(config.read_text(encoding="utf-8").replace("model: drama-3", "model: wrong"),
                          encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "drama-3 model"):
            speak("Hello.", self.home / "unused.wav")
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
