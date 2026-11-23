import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest

import torch


package = types.ModuleType("jr_script_test_package")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
core = importlib.import_module(package.__name__ + ".jr_script_core")
lib = importlib.import_module(package.__name__ + ".jr_script_library")
nodes = importlib.import_module(package.__name__ + ".jr_script")


class ScriptTests(unittest.TestCase):
    def test_schema(self):
        for cls in nodes.NODES:
            cls.define_schema().validate()

    def test_reserves_explicit_ids_and_preserves_actor(self):
        script = "王导：现在开始。\n下一场。\n<|speaker:0|> Ready? <|speaker:9|>Yes.\n15：Fine."
        text, cast, scene = core.build_script(script, scene="剧场", biographies='{"王导":"沉稳的声音"}')
        self.assertEqual([1, 1, 0, 9, 15], [x["speaker"] for x in scene["lines"]])
        self.assertEqual("剧场", scene["scene"])
        self.assertEqual("沉稳的声音", next(r for r in cast if r["name"] == "王导")["description"])
        self.assertNotIn("沉稳", text)

    def test_refuses_invalid_speakers_and_markup(self):
        for script in ("<|speaker:16|> no", "16: no", "A: <fast>tempo</slow>", "A: [未完成",
                       "A: <|im_start|>control", "\n".join("Person%d: hello" % x for x in range(17))):
            with self.subTest(script=script), self.assertRaises(ValueError):
                core.build_script(script)

    def test_chunking_preserves_scopes_phonemes_and_roles(self):
        script = "<|speaker:3|> <slow>你好世界。" + ("Repeat this " * 7) + "again.</slow> End!\n<|speaker:2|>好。"
        chunks, warnings = core.chunk_script(script, 75)
        self.assertEqual([], warnings)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c.encode()) <= 75 for c in chunks))
        for c in chunks:
            core.check_markup(c)
        self.assertTrue(all(c.startswith("<|speaker:3|>") for c in chunks[:-1]))
        self.assertTrue(chunks[-1].startswith("<|speaker:2|>"))
        def plain(text):
            return re.sub(r"\s", "", core.MARKS.sub("", text))
        self.assertEqual(plain(script), plain("".join(chunks)))
        chunks, warnings = core.chunk_script("<|speaker:3|> " + "[a really very long indivisible cue]", 32)
        self.assertTrue(warnings)

    def test_dictionary_preserves_source_and_normalizes_only_spoken_text(self):
        src = "重庆是 2026-11-23。Neil, not ONeil. [重庆 3] <|speaker:0|>"
        rules = '{"重庆":"pinyin:chong2 qing4","Neil":"arpabet:N IY1 L"}'
        model = core.normalize_numbers(core.pronunciations(src, rules), "zh")
        self.assertIn("<|phoneme_start|>chong2<|phoneme_end|><|phoneme_start|>qing4<|phoneme_end|>", model)
        self.assertIn("二零二六年十一月二十三日", model)
        self.assertIn("ONeil. [重庆 3] <|speaker:0|>", model)
        self.assertIn("<|phoneme_start|>N IY1 L<|phoneme_end|>", model)
        with self.assertRaises(ValueError):
            core.pronunciations("重庆", '{"重庆":"pinyin:chóng qìng"}')
        self.assertEqual("minus twelve point three four", core.read_number("-12.34", "en"))

    def test_rule_suggestions_preview_before_apply(self):
        text = "<|speaker:0|> 真的？\n<|speaker:1|> [sighs] yes."
        proposed = core.suggest(text, "{}")
        self.assertIn("<|speaker:0|> [questioning] 真的？", proposed)
        self.assertIn("<|speaker:1|> [sighs] yes.", proposed)
        self.assertIn("--- original", core.preview(text, proposed))

    def test_versioned_archive_and_multiple_references(self):
        wav = torch.tensor([[[0., .2, -.3, .3, 0.]]], dtype=torch.float32)
        clip, timing = lib.prepare_reference({"waveform": wav, "sample_rate": 50}, trim_edges=True)
        self.assertEqual((1, 4), (timing["start_sample"], timing["end_sample"]))
        with tempfile.TemporaryDirectory() as t:
            db = Path(t) / "voices.db"
            args = (db, "Mint", "zh", "清亮的女声", "少女", "你好", clip)
            ref_id = lib.save_reference(*args)
            self.assertEqual(ref_id, lib.save_reference(*args))
            ref2 = lib.save_reference(db, "Mint", "zh", "沉稳女声", "成年", "再见", clip)
            self.assertNotEqual(ref2, ref_id)
            self.assertTrue(torch.equal(lib.get_reference(db, ref_id)[0]["waveform"], clip["waveform"]))
            roles = [{"speaker": 0, "name": "玲", "description": "清亮少女", "audio": None, "transcript": ""}]
            assigned, info = lib.auto_cast(db, roles)
            self.assertEqual("Mint", info[0]["voice"])
            self.assertEqual(ref_id, assigned[0]["voice_id"])
            overridden, info = lib.auto_cast(db, assigned, {"玲": ref2})
            self.assertEqual(ref2, overridden[0]["voice_id"])
            self.assertEqual("override", info[0]["selection"])
            self.assertEqual(2, len(core.checked_cast([*assigned, *overridden])))


if __name__ == "__main__":
    unittest.main()
