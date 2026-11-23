import importlib
from pathlib import Path
import sys
import types
import unittest

import torch

package = types.ModuleType("jr_edit_region_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
module = importlib.import_module(package.__name__ + ".jr_edit_region")


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.source = module.audio_dict(torch.linspace(-0.8, 0.8, 44100))

    def test_video_codec_and_sample_units(self):
        for units, start, end, samples in (("video_frames", 6, 12, (11025, 22050)),
                ("samples", 123, 234, (123, 234)), ("codec_frames", 1, 2, (2048, 4096))):
            output = module.EditRegion.execute(units, start, end, 24, source=self.source)
            wave, left, right = output.result
            self.assertEqual((round(left * 44100), round(right * 44100)), samples)
            self.assertTrue(torch.equal(wave["waveform"], self.source["waveform"]))
        with self.assertRaises(ValueError):
            module.EditRegion.execute("samples", 0.5, 15, 24, source=self.source)

    def test_insert_at_exact_context_seam(self):
        a = module.audio_dict(self.source["waveform"][..., :3227])
        b = module.audio_dict(self.source["waveform"][..., 3227:])
        output = module.EditRegion.execute("join_context", 0, 0, 24, before=a, after=b)
        wave, left, right = output.result
        self.assertEqual((round(left * 44100), round(right * 44100)), (3227, 3227))
        self.assertTrue(torch.equal(wave["waveform"], self.source["waveform"]))
