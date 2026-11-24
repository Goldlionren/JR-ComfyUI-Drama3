from pathlib import Path
from contextlib import ExitStack, contextmanager
from io import BytesIO
import json
import os
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from jr_realtime_paths import MODELS, RUNTIME, model_file
from provider.server import default_db


class DeploymentPaths(unittest.TestCase):
    def test_managed_native_process_unloads_only_its_own_backend(self):
        from tools import local_service, realtime_service as service

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            state = Path(tmp)
            stack.enter_context(patch.object(local_service, "STATE", state))
            stack.enter_context(patch.object(service, "STATE", state))
            stack.enter_context(patch.object(service, "ENGINE_PORT", port))
            # Model identification uses the same config argument as audio.cpp.
            code = ("from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                    "class H(BaseHTTPRequestHandler):\n"
                    " def do_GET(self):\n"
                    "  self.send_response(200)\n"
                    "  self.end_headers()\n"
                    f"HTTPServer(('127.0.0.1', {port}),H).serve_forever()")
            args = [sys.executable, "-B", "-c", code, "--config", str(state / "realtime-5lang.json")]
            try:
                with patch.object(local_service.psutil.Process, "cmdline", return_value=[]):
                    local_service.start("realtime-engine", args, os.environ.copy(), port, "/v1/models")
                record = json.loads((state / "realtime-engine.json").read_text("utf-8"))
                self.assertEqual(record["command"], args)
                with self.assertRaisesRegex(ValueError, "resident engine"):
                    service.release_engine("q4")
                self.assertTrue(local_service.local_port(port))
                service.release_engine("5lang")
                self.assertFalse(local_service.local_port(port))
            finally:
                local_service.stop("realtime-engine")

    def test_start_after_unload_reloads_through_gateway_outside_lock(self):
        from tools import realtime_service as service

        held = False
        @contextmanager
        def lock(name):
            nonlocal held
            held = True
            try:
                yield
            finally:
                held = False

        def select(request, timeout):
            self.assertFalse(held)
            self.assertEqual(json.loads(request.data), {"model": "drama-3-q4"})
            return BytesIO(b'{"id":"drama-3-q4"}')

        with ExitStack() as stack:
            state = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(patch.object(service, "STATE", Path(state)))
            stack.enter_context(patch.object(service, "read_record", side_effect=[None, object()]))
            stack.enter_context(patch.object(service, "service_lock", side_effect=lock))
            post = stack.enter_context(patch.object(service.LOCAL_HTTP, "open", side_effect=select))
            stack.enter_context(patch("sys.argv", ["realtime_service.py", "start", "--backend", "q4"]))
            service.main()
            post.assert_called_once()

    def test_legacy_empty_process_command_requires_matching_installed_engine(self):
        from tools import local_service
        from jr_realtime_paths import engine_binary

        process = Mock()
        process.create_time.return_value = 42.0
        process.cwd.return_value = str(local_service.ROOT)
        with tempfile.TemporaryDirectory() as tmp, patch.object(local_service.psutil, "Process", return_value=process):
            state = Path(tmp)
            (state / "realtime-engine.json").write_text(json.dumps({"pid": 37, "created": 42, "command": []}), "utf-8")
            with patch.object(local_service, "STATE", state):
                process.cmdline.return_value = [str(engine_binary("5lang")), "--config", str(state / "realtime-5lang.json")]
                self.assertIs(local_service.read_record("realtime-engine"), process)
                process.cmdline.return_value = [str(engine_binary("q4")), "serve", "model.gguf", "--port", "8196"]
                self.assertIs(local_service.read_record("realtime-engine"), process)
                process.cmdline.return_value[-1] = "9000"
                self.assertIsNone(local_service.read_record("realtime-engine"))
                process.cmdline.return_value = ["some-other-process", "--port", "8196"]
                self.assertIsNone(local_service.read_record("realtime-engine"))

    def test_quantized_models_support_independent_absolute_paths(self):
        for backend in MODELS:
            variable = f"JR_DRAMA3_{backend.upper()}_MODEL"
            model = Path.cwd() / "voice models" / MODELS[backend][0]
            with patch.dict("os.environ", {variable: str(model)}):
                self.assertEqual(model_file(backend), model)
            with patch.dict("os.environ", {variable: "relative/model.gguf"}):
                with self.assertRaisesRegex(ValueError, "absolute"):
                    model_file(backend)
            with patch.dict("os.environ", {variable: ""}):
                self.assertEqual(model_file(backend), RUNTIME / MODELS[backend][0])

    def test_default_paths_and_shared_voice_database(self):
        database = Path.cwd() / "voice library" / "voices.sqlite3"
        with patch.dict("os.environ", JR_DRAMA3_VOICE_DB=str(database)):
            self.assertEqual(default_db(), database)
