import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("jr_analysis_test_package")
pkg.__path__ = [str(root)]
sys.modules[pkg.__name__] = pkg


def mod(name):
    return importlib.import_module(pkg.__name__ + "." + name)


asr = mod("jr_analysis_asr")
qa = mod("jr_analysis_quality")
video = mod("jr_analysis_video")
audio = mod("jr_delivery_audio")
nodes = mod("jr_analysis")


def tone(seconds=1.0, rate=16000):
    x = 0.3 * np.sin(2 * np.pi * 330 * np.arange(round(seconds * rate)) / rate)
    return audio.as_audio(x.astype("float32")[:, None], rate)


class LocalFileVideo:
    def __init__(self, path):
        self.path = path

    def get_stream_source(self):
        return str(self.path)


class AnalysisTests(unittest.TestCase):
    def test_schema_and_connections(self):
        registered = {c.define_schema().node_id: c for c in nodes.NODES}
        self.assertEqual(len(registered), len(nodes.NODES))
        for name in ["jr-analysis-video-dubbing", "jr-analysis-h3-lipsync"]:
            graph = json.loads((root / "example_workflows" / (name + ".json")).read_text("utf8"))
            byid = {n["id"]: n for n in graph["nodes"]}
            for idx, out, slot, inc, port, t in graph["links"]:
                self.assertEqual(byid[out]["outputs"][slot]["type"], t)
                self.assertIn(idx, byid[out]["outputs"][slot]["links"])
                self.assertEqual(byid[inc]["inputs"][port]["link"], idx)
                self.assertEqual(byid[inc]["inputs"][port]["type"], t)
            for n in byid.values():
                if n["type"] == "CreateVideo":
                    self.assertIn(n["widgets_values"][1], ("auto", 8, 10))
                if n["type"] not in registered:
                    continue
                schema = registered[n["type"]].define_schema()
                inputs = {i.id for i in schema.inputs}
                self.assertTrue(all(v["name"] in inputs for v in n["inputs"]), n["type"])
                self.assertEqual(len(n["outputs"]), len(schema.outputs))

    def test_text_diff_and_retry(self):
        a = tone()
        r = qa.review(a, "Good morning Alice.", "good morning", attempt=0, retry_limit=1)
        self.assertAlmostEqual(r["wer"], 1 / 3)
        self.assertIn("possible_truncated_tail", r["flags"])
        self.assertTrue(r["retry_requested"])
        self.assertFalse(qa.review(a, "早上好。", "早上", attempt=1, retry_limit=1)["retry_requested"])
        r = qa.review(
            a, "[笑]早上好。", "早上", alignment={"segments": [dict(start=0, end=0.5, text="早上")]}
        )
        self.assertAlmostEqual(r["cer"], 1 / 3)
        self.assertGreater(r["differences"][0]["listen_end"], 0)
        r = qa.review(a, "中文 test ONE 23", "中文 test one 23")
        self.assertEqual(r["mixed_token_error_rate"], 0)
        self.assertEqual(r["cer"], 0)

    def test_silence_and_time_validation(self):
        silent = tone()
        silent["waveform"].zero_()
        r = qa.review(silent, "Hello", "", threshold_db=-50.0, pause_s=0.5, alignment='{"segments":[]}')
        self.assertTrue(r["silent"])
        self.assertEqual(r["silences"], [{"start": 0.0, "end": 1.0}])
        self.assertEqual(r["wer"], 1)
        with self.assertRaisesRegex(ValueError, "ordered"):
            asr.ordered_segments([dict(start=0, end=0.7, text="a"), dict(start=0.5, end=1, text="b")])
        with self.assertRaisesRegex(ValueError, "longer"):
            asr.ordered_segments([dict(start=0, end=31, text="a")], max_seconds=30.0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                asr.source_path("https://example.invalid/small.pt", "NO_SUCH_ASR", [])
            f = Path(tmp) / "bad.pt"
            f.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                asr.whisper_model(str(f), "cpu")

    def test_traditional_simplified_asr_review(self):
        h = "你好 歡迎回來"
        r = qa.review(tone(), "你好，欢迎回来。", h)
        self.assertEqual(r["recognized_transcript"], h)
        self.assertGreater(r["raw_cer"], 0)
        if "unavailable" in r["chinese_normalization"]:
            self.assertEqual(r["raw_cer"], r["cer"])
        else:
            self.assertEqual(r["cer"], 0)
            self.assertFalse(r["retry_requested"])

    def test_word_ranges_translation_and_role_reference(self):
        align = json.dumps(
            {"words": [{"text": "早", "start": 0, "end": 0.4}, {"text": "上", "start": 0.4, "end": 0.8}]}
        )
        self.assertEqual(asr.selected_range(align, 0, 1), (0, 0.8, "早上"))
        self.assertEqual(asr.selected_range(align, 1, 1, manual_end=0.75), (0.4, 0.75, "上"))
        with self.assertRaises(ValueError):
            asr.selected_range(align, 1, 0)
        segments = [dict(start=0.0, end=0.8, text="Hello", speaker="Alice")]
        script, srt, matched = video.dubbing_script(segments, "你好")
        self.assertEqual(script, "Alice: 你好")
        self.assertIn("00:00:00,000 --> 00:00:00,800", srt)
        self.assertEqual(matched[0]["source_text"], "Hello")
        with self.assertRaises(ValueError):
            video.dubbing_script(segments, "你\n好")
        cast = qa.reference_cast(tone(), segments)
        self.assertEqual(cast[0]["transcript"], "Hello")
        self.assertEqual(cast[0]["audio"]["waveform"].shape[-1], 12800)

    def test_mixing_and_h3_grid(self):
        t = tone(0.2)
        bed = tone(0.7, 22050)
        bed["waveform"] = bed["waveform"].repeat(1, 2, 1)
        mix, info = video.premix(t, bed, 0, -3, 0.13)
        self.assertEqual(mix["waveform"].shape, (1, 2, 11200))
        self.assertEqual(info["start"], 0.13)
        padded, frames = video.h3_timing(t)
        self.assertEqual((frames - 5) % 17, 0)
        self.assertGreaterEqual(padded["waveform"].shape[-1], t["waveform"].shape[-1])
        self.assertEqual(padded["waveform"].shape[1], 2)
        self.assertTrue(torch.equal(t["waveform"], padded["waveform"][:, :1, :3200]))

    def test_remux_and_extract_frame_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "src.mkv"
            video.convert(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x64:rate=12:duration=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=f=440:d=1",
                    "-c:v",
                    "ffv1",
                    "-c:a",
                    "flac",
                    "-y",
                    base,
                ]
            )
            src = LocalFileVideo(base)
            orig = video.extract_track(src)
            self.assertEqual(orig["sample_rate"], 44100)
            mixed, _ = video.premix(tone(0.4), None, 0, 0, 0)
            fixed = video.remux(src, mixed, tmp)

            def frame_hash(p):
                b = video.convert(["-i", p, "-map", "0:v:0", "-f", "framemd5", "pipe:1"])
                return [
                    line.split(b",")[-1].strip()
                    for line in b.splitlines()
                    if line and not line.startswith(b"#")
                ]

            self.assertEqual(frame_hash(base), frame_hash(fixed))
            track = video.extract_track(LocalFileVideo(fixed))
            self.assertAlmostEqual(track["waveform"].shape[-1] / track["sample_rate"], 1.0, places=2)
            self.assertLess(float(track["waveform"][:, :, 8000:].abs().max()), 1e-5)


@unittest.skipUnless(os.getenv("JR_DRAMA3_ANALYSIS_REAL") == "1", "opt-in local speech-weight regression")
class ExistingLocalModels(unittest.TestCase):
    def test_actual_bidirectional_translation(self):
        for direction, src, expected in [
            ("en → zh", "The train arrives at nine tomorrow morning.", ("火车", "九")),
            ("zh → en", "明天早上九点火车到站。", ("train", "tomorrow")),
        ]:
            with self.subTest(direction=direction):
                rows = [dict(start=1.0, end=4.0, text=src, speaker="A")]
                _, _, out = video.dubbing_script(rows, translation_mode=direction)
                self.assertEqual((out[0]["start"], out[0]["end"], out[0]["speaker"]), (1.0, 4.0, "A"))
                self.assertTrue(all(s in out[0]["text"].casefold() for s in expected), out)

    def test_actual_bilingual_recognition_and_alignment(self):
        samples = [
            (asr.PARAFORMER / "example/asr_example.wav", "zh", "欢迎大家来体验达摩院推出的语音识别模型"),
            (
                Path(
                    os.getenv(
                        "JR_DRAMA3_EN_TEST_WAV", "F:/AI/IndexTTS2.5/evidence/generated/fp32/english.wav"
                    )
                ),
                "en",
                "Hello, this is a real cross-language voice cloning test.",
            ),
        ]
        for path, lang, script in samples:
            with self.subTest(language=lang):
                x, sr = sf.read(path, always_2d=True, dtype="float32")
                clip = audio.as_audio(x, sr)
                rec = asr.transcribe(clip, "Whisper small", language=lang)
                aligned = asr.align(clip, script, language=lang)
                self.assertGreaterEqual(len(aligned["words"]), 3)
                self.assertEqual(qa._tokens("".join(w["text"] for w in aligned["words"])), qa._tokens(script))
                times = [t for w in aligned["words"] for t in (w["start"], w["end"])]
                self.assertEqual(times, sorted(times))
                self.assertLessEqual(times[-1], len(x) / sr)
                quality = qa.review(clip, script, rec["text"])
                self.assertLessEqual(quality["cer"], 0.20)
                if lang == "zh":
                    p = asr.transcribe(clip, "Paraformer zh")
                    self.assertEqual(qa._tokens(p["text"]), qa._tokens(script))
                    self.assertGreater(qa.speaker_compare(clip, clip)["cosine"], 0.999)


if __name__ == "__main__":
    unittest.main()
