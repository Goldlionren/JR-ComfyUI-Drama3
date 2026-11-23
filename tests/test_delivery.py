import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import torch


package = types.ModuleType("jr_delivery_test_package")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
nodes = importlib.import_module(package.__name__ + ".jr_delivery")


def tone(duration=0.35, rate=16000, frequency=440):
    t = torch.arange(round(duration * rate), dtype=torch.float32) / rate
    return {"waveform": (0.5 * torch.sin(2 * torch.pi * frequency * t))[None, None], "sample_rate": rate}


class DeliveryTests(unittest.TestCase):
    def test_schema(self):
        for cls in nodes.NODES:
            with self.subTest(cls=cls):
                schema = cls.define_schema()
                schema.validate()
                self.assertEqual(schema.node_id, cls.__name__)
                self.assertTrue(cls.INPUT_TYPES()["required"])

    def test_srt_real_outputs_and_overflow(self):
        srt = "1\r\n00:00:00,125 --> 00:00:00,500\r\n【甲】你好\r\n\r\n2\r\n00:00:00,800 --> 00:00:01,500\r\n<v B>Hello</v>"
        rows, texts, positions, serialized, roles = nodes.JRDrama3ReadSRT.execute(srt, "").result
        self.assertEqual((texts, roles, positions), (["你好", "Hello"], ["甲", "B"], [1, 2]))
        sheet = nodes.JRDrama3CueSheet.execute(serialized)[0]
        timeline = nodes.JRDrama3CollectTakes.execute([sheet], [tone(), tone(0.45, 22050)], ["{}"])[0]
        out, srt, _ = nodes.JRDrama3MixTimeline.execute(timeline, 16000, "error", 1.3, "mix", 0, "").result
        self.assertEqual(out["waveform"].shape, (1, 1, 24000))
        self.assertEqual(float(out["waveform"][..., :2000].abs().sum()), 0)
        self.assertEqual(float(out["waveform"][..., 8010:12795].abs().sum()), 0)
        self.assertIn("00:00:00,125", srt)
        replacement = nodes.JRDrama3PutTake.execute(rows, tone(0.44), 1, "{}", timeline)[0]
        with self.assertRaisesRegex(ValueError, "exceeds its slot"):
            nodes.JRDrama3MixTimeline.execute(replacement, 16000, "error", 1.5, "mix", 0, "")
        with self.assertRaisesRegex(ValueError, "requires"):
            nodes.JRDrama3MixTimeline.execute(replacement, 16000, "time_stretch", 1.05, "mix", 0, "")
        fitted = nodes.JRDrama3MixTimeline.execute(replacement, 16000, "time_stretch", 1.5, "mix", 0, "甲")[
            0
        ]["waveform"]
        self.assertGreater(float(fitted.abs().sum()), 1)
        self.assertEqual(float(fitted[..., 8100:].abs().sum()), 0)

    def test_invalid_inputs(self):
        for bad in ["1\n00:00:01,000 --> 00:00:00,000\nNo", "1\n00:00:75,000 --> 00:00:77,000\nNo"]:
            with self.assertRaises(ValueError):
                nodes.parse_srt(bad)
        with tempfile.TemporaryDirectory() as d:
            for bad in ("../outside", r"C:\my\file.zip", r"\\server\path", "a:b"):
                with self.assertRaises(ValueError):
                    nodes.local_path(d, bad)
        with self.assertRaisesRegex(ValueError, "Expected one AUDIO"):
            nodes.JRDrama3CollectTakes.execute([[nodes.cue("1", "a"), nodes.cue("2", "b")]], [tone()], ["{}"])

    def test_join_master(self):
        joined = nodes.JRDrama3JoinAudio.execute(tone(0.4), tone(0.3, 24000), 0, 0.1)[0]
        self.assertEqual(joined["waveform"].shape[-1], 9600)
        mastered = nodes.JRDrama3MasterAudio.execute(
            joined, 0.1, 0.45, -120, 0.02, 0.02, 2, "off", -1, "24000"
        )[0]
        self.assertEqual(mastered["sample_rate"], 24000)
        self.assertEqual(mastered["waveform"].shape[-1], 8400)
        with self.assertRaisesRegex(ValueError, "gap or crossfade"):
            nodes.JRDrama3JoinAudio.execute(tone(), tone(), 0.2, 0.01)

    def test_ebook(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "一.txt").write_text("第1章 开始\n【甲】你好。\nChapter Two\nB: Bye.", encoding="utf-8")
            with patch.object(nodes.folder_paths, "get_input_directory", return_value=folder):
                rows = nodes.JRDrama3ReadBook.execute("一.txt")[0]
                self.assertEqual(
                    [(r["chapter"], r["role"]) for r in rows], [("第1章 开始", "甲"), ("Chapter Two", "B")]
                )
            with ZipFile(root / "book.epub", "w") as z:
                z.writestr(
                    "META-INF/container.xml", '<container><rootfile full-path="OPS/package.opf"/></container>'
                )
                z.writestr(
                    "OPS/package.opf",
                    '<package><manifest><item id="b" href="b.xhtml" media-type="application/xhtml+xml"/><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="a"/><itemref idref="b"/></spine></package>',
                )
                z.writestr(
                    "OPS/a.xhtml", "<html><body><h1>Opening</h1><p>A <i>little</i> line.</p></body></html>"
                )
                z.writestr("OPS/b.xhtml", "<html><body><h1>Next</h1><p>中文。</p></body></html>")
            self.assertEqual(
                [(r["chapter"], r["text"]) for r in nodes.read_book(root / "book.epub")],
                [("Opening", "A little line."), ("Next", "中文。")],
            )

    def test_bilingual_archive_chapters_and_stems(self):
        zh = [nodes.cue(1, "你好", "A", "第一章"), nodes.cue(2, "再见", "B", "第二章")]
        en = [dict(r, text=t) for r, t in zip(zh, ["Hello", "Goodbye"])]
        first = nodes.JRDrama3CollectTakes.execute([zh], [tone(), tone(0.42)], ['{"seed":5}'])[0]
        second = nodes.JRDrama3CollectTakes.execute([en], [tone(0.46), tone(0.32)], ['{"seed":6}'])[0]
        cover = torch.zeros((1, 18, 26, 3))
        cover[:, :9, :, 0] = 1
        cover[:, 9:, :, 2] = 1
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(nodes.folder_paths, "get_output_directory", return_value=folder),
        ):
            path, manifest = nodes.JRDrama3ExportProject.execute(
                first,
                "Audition",
                "zh",
                16000,
                "error",
                1.3,
                "mix",
                50,
                True,
                True,
                True,
                "mp3",
                '{"revision":"v2"}',
                second,
                cover=cover,
            ).result
            document = json.loads(manifest)
            self.assertEqual(document["languages"]["en"]["lines"][0]["settings"], {"seed": 6})
            base = (Path(folder) / path).parent
            self.assertTrue((base / "zh/mix.mp3").is_file())
            mix, _ = sf.read(base / "zh/mix.wav", dtype="float32")
            a, _ = sf.read(base / "zh/stems/001.wav", dtype="float32")
            b, _ = sf.read(base / "zh/stems/002.wav", dtype="float32")
            np.testing.assert_allclose(a + b, mix, atol=1e-6)
            chapter_info = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(base / "zh/book.m4b")]
            )
            self.assertEqual(
                [c["tags"]["title"] for c in json.loads(chapter_info)["chapters"]], ["第一章", "第二章"]
            )
            for lang in ("zh", "en"):
                book = str(base / lang / "book.m4b")
                streams = json.loads(
                    subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", book])
                )["streams"]
                self.assertTrue(any(s["disposition"].get("attached_pic") == 1 for s in streams))
                pixels = subprocess.check_output(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-i",
                        book,
                        "-map",
                        "0:v:0",
                        "-frames:v",
                        "1",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-",
                    ]
                )
                self.assertEqual(pixels, (cover[0].numpy() * 255).astype("uint8").tobytes())
            torch.testing.assert_close(
                nodes.JRDrama3RestoreProject.execute(path, "output", "first")[-1], cover
            )
            restored = nodes.JRDrama3RestoreProject.execute(path, "output", "second")[0]
            self.assertEqual(restored[0]["cue"], en[0])
            self.assertTrue(torch.equal(restored[0]["audio"]["waveform"], second[0]["audio"]["waveform"]))
            wrong = [dict(second[0], cue=dict(en[0], role="wrong"))]
            with self.assertRaisesRegex(ValueError, "same line IDs"):
                nodes.JRDrama3ExportProject.execute(
                    first,
                    "Broken",
                    "zh",
                    16000,
                    "error",
                    1.3,
                    "mix",
                    0,
                    False,
                    False,
                    False,
                    "none",
                    "{}",
                    wrong,
                )
            self.assertEqual(len(list((Path(folder) / "JR-Drama3" / "projects").iterdir())), 1)

    def test_editor_preserves_raw_takes_and_renders_trim_then_restores(self):
        rows = [nodes.cue("a", "你好", "A"), nodes.cue("b", "再见", "B")]
        report = '[{"cue":"a","seed":42},{"cue":"b","seed":43}]'
        first, second = tone(0.4), tone(0.32, frequency=220)
        timeline = nodes.JRDrama3CollectTakes.execute([rows], [first, second], [report])[0]
        cast = [dict(name="A", speaker=0, transcript="test", audio=first)]
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(nodes.folder_paths, "get_output_directory", return_value=folder),
            patch.object(nodes.folder_paths, "get_temp_directory", return_value=folder),
        ):
            init = nodes.JRDrama3TimelineEditor.execute(timeline)
            info = init.ui["jr_drama3_timeline"][0]
            self.assertTrue((Path(folder) / "jr_drama3_audition" / info["preview"]).is_file())
            edit = json.dumps(
                {
                    "a": {
                        "source": info["key"],
                        "locked": True,
                        "in_ms": 50,
                        "out_ms": 200,
                        "start_ms": 350,
                        "end_ms": 500,
                        "role": "演员",
                        "text": "新字幕",
                    }
                }
            )
            changed = nodes.JRDrama3TimelineEditor.execute(timeline, edit)[0]
            self.assertTrue(changed[0]["audio"] is first)
            mixed = nodes.JRDrama3MixTimeline.execute(changed, 16000, "error", 1.3, "mix", 0, "演员")[0][
                "waveform"
            ]
            torch.testing.assert_close(mixed[0, 0, 5600:8000], first["waveform"][0, 0, 800:3200])
            replacement = [dict(timeline[0], audio=tone(0.36, frequency=720)), timeline[1]]
            locked = nodes.JRDrama3TimelineEditor.execute(replacement, edit)[0]
            self.assertTrue(torch.equal(first["waveform"], locked[0]["audio"]["waveform"]))
            path = nodes.JRDrama3ExportProject.execute(
                changed,
                "Edit",
                "primary",
                16000,
                "error",
                1.3,
                "mix",
                0,
                False,
                False,
                False,
                "none",
                "{}",
                cast=cast,
            )[0]
            restored, _, _, _, _, loaded_cast, _, _ = nodes.JRDrama3RestoreProject.execute(
                path, "output", "first"
            ).result
            self.assertEqual(restored[0]["edit"]["in_ms"], 50)
            self.assertTrue(torch.equal(first["waveform"], restored[0]["audio"]["waveform"]))
            self.assertTrue(torch.equal(first["waveform"], loaded_cast[0]["audio"]["waveform"]))
            self.assertEqual(restored[0]["settings"]["seed"], 42)
            with self.assertRaisesRegex(ValueError, "outside"):
                nodes.JRDrama3TimelineEditor.execute(timeline, '{"a":{"out_ms":90000}}')


if __name__ == "__main__":
    unittest.main()
