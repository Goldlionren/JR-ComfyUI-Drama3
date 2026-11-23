from pathlib import Path
import unittest
from unittest.mock import patch

from jr_realtime_paths import MODELS, RUNTIME, model_file
from provider.server import default_db


class DeploymentPaths(unittest.TestCase):
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
