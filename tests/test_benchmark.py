import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("jr_benchmark_test_package")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
bench = importlib.import_module(package.__name__ + ".jr_benchmark")


class Counters:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def report(self):
        return {"sampled_peaks": {}, "scope": "test"}


class BenchmarkTests(unittest.TestCase):
    def test_registered_and_links(self):
        for node in bench.NODES:
            node.define_schema().validate()
        for filename in ("jr-benchmark-performance.json", "jr-benchmark-with-asr.json"):
            graph = json.loads((ROOT / "example_workflows" / filename).read_text(encoding="utf-8"))
            nodes = {n["id"]: n for n in graph["nodes"]}
            for link, source, output, dst, inp, kind in graph["links"]:
                self.assertIn(link, nodes[source]["outputs"][output]["links"])
                self.assertEqual(link, nodes[dst]["inputs"][inp]["link"])
                self.assertEqual(kind, nodes[source]["outputs"][output]["type"])
                self.assertEqual(kind, nodes[dst]["inputs"][inp]["type"])

    def test_reference_integrity(self):
        for key in bench.REFERENCES:
            cast, ref = bench.fixed_reference(key)
            audio = cast[0]["audio"]
            self.assertEqual(1, audio["waveform"].shape[0])
            self.assertGreater(audio["waveform"].shape[-1] / audio["sample_rate"], 3)
            self.assertGreater(float(audio["waveform"].abs().max()), .05)
            self.assertEqual(cast[0]["transcript"], ref["transcript"])

    def test_fixed_inputs_cold_warm_and_review_artifacts(self):
        events = []
        sample = {"sample_rate": 44100, "waveform": torch.ones(1, 1, 4410) * .1}

        def render(teacher, text, seed, **params):
            self.assertEqual("teacher", teacher)
            self.assertFalse(params["live"])
            self.assertTrue(params["cast"][0]["audio"]["waveform"].numel())
            events.append((text, seed))
            # Exercise the missing-stream metric for mixed speech too.
            if seed != 37003:
                params["on_audio_chunk"](sample)
            return sample, None, {"frames": 3, "finish_reason": "end"}

        clips, text, names, raw = bench.run_suite("teacher", "all", "cold_and_warm", "initial review", {},
            render, lambda: events.append("unload"), "test-model-revision", sampler=Counters)
        data = json.loads(raw)
        self.assertEqual(6, len(clips))
        self.assertEqual(3, events.count("unload"))
        self.assertEqual(events[1], events[2])
        self.assertEqual(events[4], events[5])
        self.assertEqual(events[7], events[8])
        self.assertEqual(names, [x["id"] for x in data["runs"]])
        self.assertEqual(text, [x["text"] for x in data["runs"]])
        self.assertIsNotNone(data["runs"][0]["first_stream_pcm_seconds"])
        self.assertIsNone(data["runs"][-1]["first_stream_pcm_seconds"])
        with tempfile.TemporaryDirectory() as directory:
            reviews = [json.dumps({"expected": s}) for s in text]
            body, archive = bench.finalize_report(raw, reviews, "reviewed", directory)
            self.assertEqual(json.loads(body)["runs"][2]["readback_review"]["expected"], text[2])
            self.assertEqual(Path(archive).read_text(encoding="utf-8"), body)
            self.assertEqual(json.loads(body)["listening_notes"], "reviewed")
            self.assertEqual(archive, bench.finalize_report(raw, reviews, "reviewed", directory)[1])
            with self.assertRaises(ValueError):
                bench.finalize_report(raw, reviews[:2], "incomplete", directory)


if __name__ == "__main__":
    unittest.main()
