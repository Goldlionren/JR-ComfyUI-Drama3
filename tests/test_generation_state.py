import importlib
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

package = types.ModuleType("jr_state_test_package")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
nodes = importlib.import_module(package.__name__ + ".jr_generate")
teacher = importlib.import_module(package.__name__ + ".jr_teacher")
jobs = importlib.import_module(package.__name__ + ".jr_jobs")
runtime = importlib.import_module(package.__name__ + ".jr_runtime")


def take(text="Hello.", seed=1, reason="eos"):
    audio = {"waveform": torch.ones(1, 1, 32), "sample_rate": 44100}
    return audio, {"utterance": text}, {"seed": seed, "mode": "line", "finish_reason": reason}


class GenerationStateTests(unittest.TestCase):
    def test_fla_toggle_preserves_legacy_take_ids_and_separates_cached_audio(self):
        with TemporaryDirectory() as folder, patch.object(nodes, "_take_dir", return_value=Path(folder)), \
                patch.object(nodes, "validate_teacher", return_value=Path(folder)), \
                patch.object(nodes, "render", return_value=take()) as render:
            legacy = nodes.Teacher.execute(folder)[0]
            fast = nodes.Teacher.execute(folder, use_fla=True)[0]
            native = nodes.Teacher.execute(folder, use_fla=False)[0]
            self.assertEqual(Path(fast), Path(legacy))
            def candidate(selection):
                return nodes.Candidates.execute(selection, "Hello.", 1, 1, "")[0][0]
            old_key = candidate(legacy)
            fast_key = candidate(fast)
            self.assertNotEqual(old_key, fast_key)
            self.assertEqual(candidate(native), old_key)
            self.assertEqual(candidate(fast), fast_key)
            self.assertEqual(render.call_count, 2)

    def test_legacy_truncated_take_is_never_selected_and_is_replaced_only_on_success(self):
        with TemporaryDirectory() as directory, patch.object(nodes, "_take_dir", return_value=Path(directory)):
            identity = dict(format=1, revision=teacher.REVISION, text="Hello.", seed=1,
                            cast=jobs.cast_key([]), mode="line", direction="", settings=nodes.DEFAULTS)
            key = jobs.key_for(identity)
            bad = take(reason="length")
            with self.assertRaises(jobs.IncompleteTakeError):
                jobs.save_take(directory, key, bad)
            path = Path(directory) / (key + ".safetensors")
            self.assertFalse(path.exists())
            meta = jobs.as_json(dict(key=key, sample_rate=44100, history=bad[1], report=bad[2]))
            save_file({"waveform": bad[0]["waveform"]}, str(path), metadata={"take": meta})
            self.assertIsNone(jobs.read_take(directory, key))
            with self.assertRaises(FileNotFoundError):
                nodes.SelectTake.execute(1, key)
            with patch.object(nodes, "render", side_effect=[bad, take()]):
                with self.assertRaises(jobs.IncompleteTakeError):
                    nodes.render_cues("model", [{"text": "Hello."}], 1, "linked", [], None)
                self.assertIsNone(jobs.read_take(directory, key))
                nodes.render_cues("model", [{"text": "Hello."}], 1, "linked", [], None)
            self.assertEqual(nodes.SelectTake.execute(1, key)[1], {"utterance": "Hello."})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_candidates_keep_slot_numbers_and_reuse_independent_batch_and_design(self):
        def generate(directory, text, seed, **options):
            return take(text, seed, "length" if seed == 7 else "eos")
        with TemporaryDirectory() as directory, patch.object(nodes, "_take_dir", return_value=Path(directory)), \
                patch.object(nodes, "render", side_effect=generate) as render:
            nodes.render_cues("model", [{"text": "Hello."}], 6, "independent", [], None)
            keys, reports = nodes.Candidates.execute("model", "Hello.", 6, 3, "").result
            self.assertEqual(render.call_count, 3)
            self.assertEqual([item["candidate"] for item in json.loads(reports)], [1, 2, 3])
            self.assertIsNone(keys[1])
            with self.assertRaises(jobs.IncompleteTakeError):
                nodes.SelectTake.execute(2, takes=keys)
            self.assertEqual(nodes.SelectTake.execute(3, takes=keys)[2], keys[2])
            self.assertEqual(len(list(Path(directory).glob("*.safetensors"))), 2)
            line_key = nodes.Candidates.execute("model", "Hello.", 6, 1, "warm")[0][0]
            design_key = nodes.Candidates.execute("model", "Hello.", 6, 1, "warm", mode="design")[0][0]
            self.assertNotEqual(line_key, design_key)
            self.assertEqual(render.call_args.kwargs["mode"], "design")
            self.assertEqual(render.call_args.kwargs["direction"], "warm")
            nodes.Candidates.execute("model", "Hello.", 6, 1, "warm", mode="design")
            self.assertEqual(render.call_count, 5)
            with self.assertRaises(ValueError):
                nodes.Candidates.execute("model", "Hello.", 6, 1, "", mode="design")

    def test_checked_take_skips_truncation_before_asr_and_exhausts_retry_limit(self):
        asr = importlib.import_module(package.__name__ + ".jr_analysis_asr")
        quality = importlib.import_module(package.__name__ + ".jr_analysis_quality")
        with TemporaryDirectory() as directory, patch.object(nodes, "_take_dir", return_value=Path(directory)), \
                patch.object(nodes, "render", side_effect=[take(reason="length"), take(seed=9)]) as render, \
                patch.object(asr, "transcribe", return_value={"text": "Hello."}) as hear, \
                patch.object(quality, "review", return_value={"flags": [], "cer": 0.}):
            _, _, keys, report = nodes.CheckedTake.execute("model", "Hello.", "", "en", 8, 1, 0.1).result
            self.assertEqual(len(keys), 1)
            self.assertEqual(len(json.loads(report)["attempts"]), 2)
            self.assertEqual(hear.call_count, 1)
            self.assertEqual(render.call_count, 2)
            with patch.object(nodes, "render", return_value=take(reason="context_limit")):
                with self.assertRaises(jobs.IncompleteTakeError):
                    nodes.CheckedTake.execute("model", "Another.", "", "en", 17, 0, 0.1)
            self.assertEqual(hear.call_count, 1)

    def test_batch_reserves_explicit_speakers_and_preserves_legacy_independent_key(self):
        script = [{"role": "Alice", "text": "Hi."}, {"role": "Bo", "text": "<|speaker:0|>Hello."},
                  {"role": "BO", "text": "Bye."}]
        def generate(directory, text, seed, **options):
            return take(text, seed)
        with TemporaryDirectory() as directory, patch.object(nodes, "_take_dir", return_value=Path(directory)), \
                patch.object(nodes, "render", side_effect=generate) as render:
            identity = dict(format=1, revision=teacher.REVISION, text="<|speaker:1|>Hi.", seed=11,
                            cast=jobs.cast_key([]), mode="line", source=None, settings=nodes.DEFAULTS)
            key = jobs.key_for(identity)
            jobs.save_take(directory, key, take(identity["text"], 11))
            result = nodes.render_cues("model", script, 11, "independent", [], None)
            self.assertEqual(json.loads(result[2])[0]["take"], key)
            self.assertEqual([call.args[1] for call in render.call_args_list],
                             ["<|speaker:0|>Hello.", "<|speaker:0|>Bye."])
            nodes.Candidates.execute("model", identity["text"], 11, 1, "")
            self.assertEqual(render.call_count, 2)

    def test_second_segment_truncation_stops_unfinished_take_without_decoding_it(self):
        part = torch.ones((10, 1), dtype=torch.long)
        samples = [([part], "eos", [], []), ([part], "length", [], [])]
        with torch.random.fork_rng(devices=[]), \
                patch.object(runtime.torch.cuda, "is_available", return_value=True), \
                patch.object(runtime.torch.cuda, "device", return_value=nullcontext()), \
                patch.object(runtime.torch.cuda, "default_generators", [Mock()]), \
                patch.object(runtime.torch.random, "fork_rng", return_value=nullcontext()), \
                patch.object(runtime, "_get_engine", return_value=(None, None, 0)), \
                patch.object(runtime, "split_line", return_value=["One.", "Two.", "Three."]), \
                patch.object(runtime, "render_line", side_effect=lambda text, direction: text), \
                patch.object(runtime, "build_prompt", return_value=torch.zeros((1, 11), dtype=torch.long)), \
                patch.object(runtime, "sample", side_effect=samples) as sample, \
                patch.object(runtime, "decode", return_value=torch.ones(32)) as decode:
            with self.assertRaisesRegex(runtime.UnfinishedSpeechError, "2/3.*length"):
                runtime.synthesize(".", "One. Two. Three.", device="cuda:0")
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(decode.call_count, 1)

    def test_live_truncation_never_finishes_a_take_or_reads_beyond_lookahead(self):
        seen = []
        def chunks():
            for text in ("one", "two", "three"):
                seen.append(text)
                yield text
        notify = Mock()
        with patch.object(runtime, "synthesize", return_value={"finish_reason": "context_limit"}):
            with self.assertRaisesRegex(runtime.UnfinishedSpeechError, "context_limit"):
                runtime.synthesize_stream(".", chunks(), on_take=notify)
        self.assertEqual(seen, ["one", "two"])
        notify.assert_not_called()

    def test_history_modes_and_bounded_checkpoint_for_long_books(self):
        old = [{"user": str(i), "parts": [[[i]] * 10]} for i in range(9)]
        refs = [{"speaker": 0, "text": "Ref", "codes": [[1]] * 10}]
        previous = {"schema": 1, "turns": old, "references": refs}
        part = torch.ones((10, 1), dtype=torch.long)
        with torch.random.fork_rng(devices=[]), \
                patch.object(runtime.torch.cuda, "is_available", return_value=True), \
                patch.object(runtime.torch.cuda, "device", return_value=nullcontext()), \
                patch.object(runtime.torch.cuda, "default_generators", [Mock()]), \
                patch.object(runtime.torch.random, "fork_rng", return_value=nullcontext()), \
                patch.object(runtime, "_get_engine", return_value=(None, None, 0)), \
                patch.object(runtime, "checked", side_effect=lambda text, field: text), \
                patch.object(runtime, "split_line", side_effect=lambda text: [text]), \
                patch.object(runtime, "render_line", side_effect=lambda text, direction: text), \
                patch.object(runtime, "build_prompt", return_value=torch.zeros((1, 11), dtype=torch.long)) as build, \
                patch.object(runtime, "sample", return_value=([part], "eos", [], [])), \
                patch.object(runtime, "decode", return_value=torch.ones(32)):
            for mode in ("line", "design", "continue", "revise", "stream"):
                text = {"current": "Now", "lookahead": ""} if mode == "stream" else "Now"
                out = runtime.synthesize(".", text, previous=previous, mode=mode, direction="calm")
                _, system, references, context, _ = build.call_args.args
                if mode in ("line", "design"):
                    self.assertEqual(context, [])
                    self.assertEqual(references, [])
                    self.assertEqual(len(out["history"]["turns"]), 1)
                else:
                    self.assertEqual(context, [old[0], *old[-2:]])
                    self.assertEqual(references, refs)
                    self.assertEqual(out["history"]["turns"][:2], [old[0], old[-1]])
                    self.assertEqual(len(out["history"]["turns"]), 3)
                if mode == "design":
                    self.assertIn("Characters:\n- Narrator: calm", system)
            self.assertEqual(len(previous["turns"]), 9)

    def test_sample_accepts_eos_immediately_after_last_allowed_frame(self):
        runtime_model = importlib.import_module(package.__name__ + ".jr_runtime.model")
        if not (runtime_model.runtime_dir() / "fish-speech").is_dir():
            self.skipTest("Requires pinned Fish runtime.")
        runtime_model.activate_runtime()
        inference = importlib.import_module(package.__name__ + ".jr_runtime.inference")
        tower = Mock(device=torch.device("cpu"), semantic_begin=10, semantic_end=11)
        tower.embed_backbone.side_effect = lambda rows, embed: torch.zeros((len(rows), 1))
        backbone = Mock(side_effect=lambda inputs_embeds, input_pos: inputs_embeds)
        tokenizer = Mock()
        tokenizer.token_to_id.return_value = 4
        tokenizer.encode.return_value.ids = [3]
        model = types.SimpleNamespace(backbone=backbone, audio_decoder=tower, tokenizer=tokenizer,
                  config=types.SimpleNamespace(eos_token_id=2), cache_capacity=32)
        prompt = torch.zeros((1, 11), dtype=torch.long)
        for last, expected in ((2, "eos"), (10, "length")):
            tower.decode_step.side_effect = [
                torch.tensor([[token] + [0] * 10], dtype=torch.long) for token in (10, last)]
            notify = Mock()
            parts, reason, _, _ = inference.sample(model, prompt, 1, 0.7, 0.9, 0, 0, None, notify)
            self.assertEqual(reason, expected)
            self.assertEqual(parts[0].shape, (10, 1))
            self.assertEqual(notify.call_count, 1)

    def test_legacy_and_mixed_model_rejected(self):
        with TemporaryDirectory() as work:
            (Path(work) / "model-00001-of-00008.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "八分片"):
                teacher.validate_teacher(work)
        self.assertEqual(sum(n.endswith(".safetensors") for n in teacher.MANIFEST["files"]), 15)

    def test_retry_preserves_completed_lines(self):
        calls = []
        def generate(directory, text, seed, **options):
            calls.append((text, options.get("previous")))
            if len(calls) == 2:
                raise InterruptedError()
            audio = {"waveform":torch.ones(1,1,32), "sample_rate":44100}
            return audio, {"utterance": text}, {"seed": seed, "finish_reason": "eos"}
        cues = [{"role":"A","text":"One."},{"role":"B","text":"Two."}]
        with TemporaryDirectory() as directory, patch.object(nodes,"_take_dir",return_value=Path(directory)), \
                patch.object(nodes,"render",side_effect=generate):
            with self.assertRaises(InterruptedError):
                nodes.render_cues("model", cues, 1, "linked", [], None)
            out = nodes.render_cues("model", cues, 1, "linked", [], None)
            self.assertEqual(len(out[0]), 2)
            self.assertEqual(len(calls), 3)
            self.assertTrue(calls[0][0].startswith("<|speaker:0|>"))
            self.assertEqual(calls[2][1], {"utterance":calls[0][0]})
            cues[0]["text"]="Changed."
            nodes.render_cues("model", cues, 1, "linked", [], None)
            self.assertEqual(len(calls), 5)

    def test_all_schemas_and_reference_fingerprints(self):
        live = importlib.import_module(package.__name__+".jr_live")
        for cls in [*nodes.NODES, live.LiveText]:
            cls.define_schema().validate()
        a={"waveform":torch.zeros(1,1,64),"sample_rate":44100}
        first=jobs.cast_key([{"audio":a}])
        a["waveform"][0,0,1]=1
        self.assertNotEqual(first,jobs.cast_key([{"audio":a}]))
        with self.assertRaises(ValueError):
            nodes.SelectTake.execute(1, "../escape")
