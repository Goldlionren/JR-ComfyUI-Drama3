from pathlib import Path
import unittest
from unittest.mock import patch
import sys

from provider.server import default_db
from tools import local_service

# The realtime launcher is also executable as a script from tools/.
with patch.dict(sys.modules, local_service=local_service):
    from tools import realtime_service


class DeploymentPaths(unittest.TestCase):
    def test_nvfp4_external_file_and_q4_default(self):
        model = Path.cwd() / "voice models" / realtime_service.MODELS["nvfp4"][0]
        with patch.dict("os.environ", JR_DRAMA3_NVFP4_MODEL=str(model)):
            self.assertEqual(realtime_service.model_file("nvfp4"), model)
            self.assertEqual(realtime_service.model_file("q4"),
                             realtime_service.RUNTIME / realtime_service.MODELS["q4"][0])
        with patch.dict("os.environ", JR_DRAMA3_NVFP4_MODEL="relative/model.gguf"):
            with self.assertRaisesRegex(ValueError, "absolute"):
                realtime_service.model_file("nvfp4")

    def test_default_paths_and_shared_voice_database(self):
        with patch.dict("os.environ", JR_DRAMA3_NVFP4_MODEL=""):
            self.assertEqual(realtime_service.model_file("nvfp4"),
                             realtime_service.RUNTIME / realtime_service.MODELS["nvfp4"][0])
        database = Path.cwd() / "voice library" / "voices.sqlite3"
        with patch.dict("os.environ", JR_DRAMA3_VOICE_DB=str(database)):
            self.assertEqual(default_db(), database)
