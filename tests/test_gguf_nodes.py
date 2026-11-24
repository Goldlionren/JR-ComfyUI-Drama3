import asyncio
from contextlib import ExitStack
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from provider.realtime import native_sampling
from provider.realtime_voices import decode_previous, previous_options, voice_options


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jr_node_audit", root / "__init__.py",
                                           submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = importlib.import_module(spec.name + ".jr_generate")
quant = importlib.import_module(spec.name + ".jr_gguf")
benchmark = importlib.import_module(spec.name + ".jr_benchmark")
script = importlib.import_module(spec.name + ".jr_script")


class QuantizedNodeTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("JR_DRAMA3_TEST_GGUF"), "requires a running GGUF gateway")
    def test_installed_gguf_production_nodes(self):
        """Run against the requested local engine with an isolated candidate directory."""
        import folder_paths
        live = importlib.import_module(spec.name + ".jr_realtime")
        model = nodes.Teacher.execute(model=os.environ["JR_DRAMA3_TEST_GGUF"])[0]
        self.assertIsInstance(model, quant.QuantizedModel)
        cast, _ = benchmark.fixed_reference("en")
        with TemporaryDirectory() as tmp, patch.object(nodes, "_take_dir", return_value=Path(tmp)), \
                patch.object(folder_paths, "temp_directory", tmp):
            first = nodes.Generate.execute(model, "Hello.", "line", "", 184, False, cast=cast)
            self.assertEqual(json.loads(first[2])["revision"], model.sha256)
            takes = nodes.Candidates.execute(model, "Hi.", 185, 1, "", cast=cast)[0]
            clips = [first[0], nodes.SelectTake.execute(1, takes=takes)[0]]
            clips.extend(nodes.Batch.execute(model, [{"text": "Hello."}, {"text": "Hi."}], 184, "auto", cast)[0])
            clips.append(nodes.LongForm.execute(model, "Hello.", 184, 300, 0, cast)[0])
            preset_lines = {"en": "Hello.", "zh": "你好。"}
            if model.id == "drama-3-5lang":
                preset_lines.update(ja="こんにちは。", ko="안녕하세요.", yue="早晨，你好。")
            for voice, line in preset_lines.items():
                clips.append(live.RealtimeSpeech.PREPARE_CLASS_CLONE(None).execute(line, voice, 184, False, model=model.id)[0])
            clips.append(nodes.Generate.execute(model, "Good morning.", "design", "calm", 184, False)[0])
            if model.id != "drama-3-q4":
                for mode in ("continue", "revise"):
                    clips.append(nodes.Generate.execute(model, "Good morning.", mode, "", 184, False, previous=first[1])[0])
            for audio in clips:
                wave = audio["waveform"]
                self.assertEqual(audio["sample_rate"], 44100)
                self.assertEqual(wave.shape[:2], (1, 1))
                self.assertGreater(wave.shape[-1], 1000)
                self.assertTrue(torch.isfinite(wave).all())
                self.assertGreater(wave.abs().max().item(), .001)

    def test_cast_library_and_script_feed_the_selected_model(self):
        cast, _ = benchmark.fixed_reference("en")
        ref = cast[0]
        audio = script.JRDrama3TrimReference.execute(ref["audio"], 0, 0, False, -50, False)[0]
        with TemporaryDirectory() as directory, patch.object(script, "voice_db", return_value=Path(directory) / "声音.sqlite3"):
            voice = script.JRDrama3SaveVoice.execute(audio, "Alex", ref["transcript"], "calm", "en", "dialogue")[0]
            self.assertEqual(script.JRDrama3FindVoice.execute(voice, "en", 1)[2], voice)
            roles = script.JRDrama3Reference.execute("Alex", 2, "", "calm", "en")[0]
            assigned = script.JRDrama3AutoCast.execute(roles, json.dumps({"Alex": voice}))[0]
            text, bound, _ = script.JRDrama3Script.execute("Alex: Hi.", "", "{}", assigned).result
            directed = script.JRDrama3Direct.execute(text, "quietly")[0]
            text = script.JRDrama3Pronunciation.execute(directed, "{}", "off")[0]
            text = script.JRDrama3SuggestCues.execute(text, "{}", False, "")[0]
            line = script.JRDrama3Chunks.execute(text, 300)[0][0]
            settings = nodes.Sampling.execute(.6, .9, 13, 90)[0]
            def speech(*args, **kwargs):
                yield b"\x00\x20" * 100

            with patch.object(quant, "stream_speech", side_effect=speech) as calls:
                result = nodes.Generate.execute(quant.QuantizedModel("drama-3-q4", "a"*64),
                                                line, "line", "", 2, False, cast=bound, settings=settings)
            self.assertIn("Hi.", calls.call_args.args[0])
            self.assertEqual(calls.call_args.kwargs["reference_text"], ref["transcript"])
            self.assertEqual(calls.call_args.kwargs["sampling"], settings)
            self.assertEqual(json.loads(result[2])["segments"][0]["speaker"], 2)

    def test_every_registered_schema_and_published_workflow(self):
        async def schemas():
            ext = await package.comfy_entrypoint()
            return [node.GET_SCHEMA() for node in await ext.get_node_list()]
        found = asyncio.run(schemas())
        ids = {s.node_id for s in found}
        self.assertEqual(len(found), len(ids))
        reviewed = (root / "docs" / "功能与工作流.md").read_text("utf-8")
        for node_id in ids:
            self.assertIn(f"| `{node_id.removeprefix('JRDrama3')}` ", reviewed)
        for schema in found:
            with self.subTest(node=schema.node_id):
                schema.validate()
        for file in (root / "example_workflows").glob("*.json"):
            graph = json.loads(file.read_text("utf-8"))
            for n in graph.get("nodes", []):
                if n["type"].startswith("JRDrama3"):
                    self.assertIn(n["type"], ids, file)
        schema = nodes.Teacher.define_schema()
        self.assertEqual([i.id for i in schema.inputs], ["directory", "use_fla", "model"])
        self.assertEqual(schema.inputs[-1].default, "FP8")
        self.assertEqual(set(schema.inputs[-1].options), {"FP8", *nodes.PUBLIC_MODELS})

    def test_saved_candidates_are_separate_by_backend_and_weight_digest(self):
        audio = {"waveform": torch.ones(1, 1, 128) * .2, "sample_rate": 44100}
        cast = [{"speaker": 2, "name": "Ann", "audio": audio, "transcript": "Reference."}]
        calls = []
        def speech(line, voice, seed, **kwargs):
            calls.append((line, voice, seed, kwargs))
            yield b"\x00\x20" * 100
        keys = set()
        with ExitStack() as stack:
            folder = stack.enter_context(TemporaryDirectory())
            stack.enter_context(patch.object(nodes, "_take_dir", return_value=Path(folder)))
            stack.enter_context(patch.object(quant, "stream_speech", side_effect=speech))
            stack.enter_context(patch.object(nodes, "validate_teacher", side_effect=AssertionError("loaded FP8")))
            for model, digest in [("drama-3-q4", "1"), ("drama-3-5lang", "2"),
                                  ("drama-3-nvfp4", "3"), ("drama-3-5lang", "4")]:
                with patch.object(nodes, "installed_model", return_value=digest * 64):
                    selected = nodes.Teacher.execute(model=model)[0]
                    fingerprint = nodes.Teacher.fingerprint_inputs(model=model)
                self.assertEqual(fingerprint, (model, digest * 64))
                for _ in range(2):
                    takes = nodes.Candidates.execute(selected, "<|speaker:2|>Hi.", 27, 1, "", cast)[0]
                keys.add(takes[0])
                restored = nodes.SelectTake.execute(1, takes=takes)
                self.assertEqual(restored[0]["waveform"].shape, (1, 1, 100))
                self.assertEqual(calls[-1][:3], ("Hi.", "zh", 27))
                self.assertEqual(calls[-1][3]["reference_text"], "Reference.")
                self.assertEqual(calls[-1][3]["model"], model)
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(calls), 4)

    def test_linked_history_is_acoustic_and_bound_to_selected_model(self):
        def stub(*args, **kwargs):
            yield b"\x00\x20" * 100
        with patch.object(quant, "stream_speech", side_effect=stub) as speech:
            model = quant.QuantizedModel("drama-3-5lang", "a" * 64)
            history = nodes.Generate.execute(model, "First.", "line", "", 1, False)[1]
            nodes.Generate.execute(model, "Next.", "continue", "", 2, False, previous=history)
            self.assertEqual(speech.call_args.kwargs["previous_take"]["text"], "First.")
            decoded = decode_previous(speech.call_args.kwargs["previous_take"], "5lang")
            with previous_options(decoded) as native:
                record, = json.loads(native["previous_takes"])
                path = Path(record["audio_path"])
                self.assertEqual(path.read_bytes()[:4], b"RIFF")
            self.assertFalse(path.exists())
            other = quant.QuantizedModel("drama-3-nvfp4", "b" * 64)
            with self.assertRaisesRegex(ValueError, "同一 GGUF"):
                nodes.Generate.execute(other, "Next.", "revise", "", 2, False, previous=history)
            q4 = quant.QuantizedModel("drama-3-q4", "c" * 64)
            with patch.object(nodes, "unload_model") as release:
                report, audio = nodes.Unload.execute(teacher=q4).result
            release.assert_called_once_with(q4.id)
            self.assertIn(q4.id, report)
            self.assertIsNone(audio)
            with self.assertRaisesRegex(ValueError, "Q4"):
                nodes.Generate.execute(q4, "Next.", "continue", "", 2, False, previous=history)
            with self.assertRaisesRegex(ValueError, "independent"):
                nodes.Batch.execute(q4, [{"text": "First."}], 1, "linked")
            with self.assertRaisesRegex(ValueError, "infill"):
                nodes.Infill.execute(model, {"waveform": torch.ones(1, 1, 100), "sample_rate": 100},
                                     "New line.", 0, 0.5, 1)
            with patch.object(benchmark, "run_suite", return_value=([], [], [], "{}")) as run:
                benchmark.JRDrama3Benchmark.execute(q4, "en", "cold_and_warm", "")
            self.assertEqual(run.call_args.args[-1], q4.id + "@" + q4.sha256)

    def test_sampling_and_voice_mapping_on_all_native_engines(self):
        for backend in ("5lang", "nvfp4", "q4"):
            opts = native_sampling(nodes.DEFAULTS, backend)
            self.assertEqual(opts["top_k"], 4097)
            self.assertEqual(opts["top_p"], .9)
            if backend == "q4":
                self.assertAlmostEqual(opts["max_seconds"], 1024 * 2048 / 44100)
                self.assertEqual(voice_options({"voice": "none"}, backend), {"voice": "none"})
            else:
                self.assertEqual(opts["max_tokens"], 1024)
                self.assertEqual(voice_options({"voice": "none"}, backend), {})
        for bad in [{"arbitrary": 1}, {"temperature": float("nan")}, {"max_frames": 20.5}, {"top_k": True}]:
            with self.assertRaises(ValueError):
                native_sampling(bad, "q4")
        with self.assertRaises(ValueError):
            decode_previous({"text": "Hi.", "audio_path": "/private/file"}, "5lang")
