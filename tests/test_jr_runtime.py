import unittest
from unittest.mock import Mock, patch
import sys
from types import SimpleNamespace

import torch

import jr_runtime as runtime
from jr_runtime import inference, model, prompt


class RuntimeTests(unittest.TestCase):
    def test_cold_loading_requests_only_required_space(self):
        manager = Mock()
        device = torch.device("cuda:0")
        with patch.dict(sys.modules, {"comfy.model_management": manager}), \
                patch.object(torch.cuda, "empty_cache") as release, \
                patch.object(torch.cuda, "mem_get_info", side_effect=[(28 * model.GIB, 32 * model.GIB),
                                                                     (7 * model.GIB, 32 * model.GIB)]):
            model.make_room_for_teacher(device)
            manager.free_memory.assert_not_called()
            model.make_room_for_teacher(device)
            manager.free_memory.assert_called_once_with(13 * model.GIB, device)
            release.assert_called_once()

    def test_mono_audio_preserves_samples_and_validates_batches(self):
        source = torch.linspace(-1, 1, 64).reshape(1, 1, -1)
        assert torch.equal(inference.mono_audio({"waveform": source, "sample_rate": 44100}), source.flatten())
        with self.assertRaises(ValueError):
            inference.mono_audio({"waveform": source.expand(2, 1, -1), "sample_rate": 44100})


    def test_incremental_text_reads_only_one_chunk_ahead(self):
        read, completed = [], []
        def chunks():
            for word in ("one", "two", "three"):
                read.append(word)
                yield word
        def synth(model_dir, text, refs, seed, **kwargs):
            index = len(completed)
            assert kwargs["mode"] == "stream"
            assert kwargs["use_fla"]
            assert text["current"] == ("one", "two", "three")[index]
            assert text["lookahead"] == ("two", "three", "")[index]
            assert len(read) == min(index + 2, 3)
            history = kwargs["previous"] or {"turns": []}
            return {"audio": inference.audio_dict(torch.ones(16)), "history": {"turns": [*history["turns"], text]},
                    "frames": 2, "finish_reason": "eos", "host_expert_layers": 0}
        with patch.object(runtime, "synthesize", synth):
            result = runtime.synthesize_stream(".", chunks(), use_fla=True, on_take=lambda i, take: completed.append(i))
        assert result["audio"]["waveform"].shape == (1, 1, 48)
        assert [s["start_sample"] for s in result["segments"]] == [0, 16, 32]
        assert len(result["history"]["turns"]) == 3


    def test_reference_codes_reject_invalid_shapes_or_ranges(self):
        assert prompt.codes_tensor([[100] * 6] * 10).shape == (10, 6)
        for invalid in ([], [[1, 1]], [[4096]] * 10, [[1.1]] * 10, [[True]] * 10):
            with self.assertRaises(ValueError):
                prompt.codes_tensor(invalid)

    def test_phonemes_are_paired_in_long_clips(self):
        if not (model.runtime_dir() / "fish-speech").is_dir():
            self.skipTest("Requires pinned Fish runtime.")
        model.activate_runtime()
        syllable = "<|phoneme_start|>chong2<|phoneme_end|>"
        script = "中" * 93 + syllable + "文" * 50
        assert prompt.checked(script, "script") == script
        chunks = prompt.split_line(script)
        assert len(chunks) > 1
        assert sum(syllable in chunk for chunk in chunks) == 1
        assert all(prompt.checked(chunk, "chunk") == chunk for chunk in chunks)
        nested = syllable.replace("chong2", "a<|speaker:1|>b")
        for bad in (syllable.replace("<|phoneme_end|>", ""), "<|phoneme_end|>", "<|fim_prefix|>", nested):
            with self.assertRaises(ValueError):
                prompt.checked(bad, "script")

    def test_scene_separates_names_and_non_acoustic_boundary_rows(self):
        if not (model.runtime_dir() / "fish-speech").is_dir():
            self.skipTest("Requires pinned Fish runtime.")
        model.activate_runtime()
        content = {"characters": [{"speaker": 0, "name": "Lin", "description": "gentle"}],
                   "lines": [{"speaker": 0, "name": "Lin", "text": "Hello."}]}
        system, script, _ = prompt.parse_scene(content, [], "")
        assert "Lin" in system and "Lin" not in script and "<|speaker:0|>" in script
        # Select an actual newline from a miniature logit head. Neither
        # choosing a newline nor redirecting an early EOS should sample audio.
        tower = Mock()
        tower.main_candidates = torch.tensor([1, 2, 3, 4])
        tower.semantic_begin, tower.semantic_end, tower.eos_id = 3, 4, 2
        tower.is_semantic.side_effect = lambda token: (token >= 3) & (token <= 4)
        for raw, redirect in ((1, False), (2, True)):
            logits = torch.zeros(1, 5)
            logits[0, raw] = 100
            row, token, corrected = inference.scene_frame(
                tower, logits, torch.zeros(1, 1), torch.tensor([0]), torch.zeros(1, 11, dtype=torch.long),
                torch.tensor([0]), temperature=0.7, top_p=0.9, top_k=1, newline=1, early_eos=redirect)
            assert token == 1 and corrected == redirect and row[0, 1:].sum() == 0
        assert tower.prefill.call_count == 2
        tower.run_shared.assert_not_called()

    def test_infill_supplies_only_unchanged_context(self):
        if not (model.runtime_dir() / "fish-speech").is_dir():
            self.skipTest("Requires pinned Fish runtime.")
        model.activate_runtime()
        # Unique sample values let us check exact source windows fed to the codec.
        hop = inference.HOP
        source = torch.arange(10 * hop, dtype=torch.float32)
        separate = []
        def encode_context(codec, wave, device):
            separate.append(wave)
            return torch.ones((10, (wave.numel() + hop - 1) // hop), dtype=torch.long)
        edit = {"audio": inference.audio_dict(source), "start_seconds": (3 * hop + 12) / inference.RATE,
                "end_seconds": (6 * hop + 31) / inference.RATE}
        with patch.object(runtime, "encode", encode_context):
            fields, (_, left, right, _) = runtime._edit_inputs(None, "cpu", "Revised line.", edit)
        assert (left, right) == (3 * hop + 12, 6 * hop + 31)
        assert torch.equal(separate[0], source[:3*hop])
        assert torch.equal(separate[1], source[7*hop:])
        assert fields[1] == "<|fim_suffix|>" and fields[3] == "<|fim_prefix|>"
        assert fields[2].shape[1] == 3 and fields[4].shape[1] == 3

    def test_official_prompt_parity(self):
        directory = model.runtime_dir()
        checkpoint = directory / "teacher"
        if not (checkpoint / "tokenizer.json").exists() or not (directory / "fish-speech").is_dir():
            self.skipTest("Requires local teacher tokenizer and pinned Fish source.")
        import json
        from tokenizers import Tokenizer
        model.activate_runtime()
        from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
        from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
        # Re-entering after the namespace was imported must still select the pinned source.
        model.activate_runtime()
        config = FishOmniConfig.from_dict(json.loads((checkpoint / "config.json").read_text("utf-8")))
        tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        prompt.verify_control_tokens(tokenizer)
        engine = SimpleNamespace(tokenizer=tokenizer, config=config)
        transcript = "你好，World!"
        user = prompt.render_line(transcript, "")
        codes = torch.arange(30).reshape(10, 3)
        refs = [{"speaker": 4, "text": transcript, "codes": codes}]
        actual = prompt.build_prompt(engine, "convert the provided text to speech reference to the following:", refs, [], [user])
        expected = build_drama3_prompt(transcript, tokenizer, config, [f"<|speaker:4|> {transcript}"], [codes])
        assert torch.equal(actual, expected)
