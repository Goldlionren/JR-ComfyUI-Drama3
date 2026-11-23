"""ComfyUI nodes for local Drama 3 timeline editing and delivery."""

import json

import folder_paths
import numpy as np
from comfy_api.latest import io

from .jr_delivery_audio import (
    arrange,
    as_audio,
    blocks,
    mix_gain,
    placed_rows,
    postprocess,
    resample,
    samples,
)
from .jr_delivery_bundle import export_project, restore_project
from .jr_delivery_editor import JRDrama3TimelineEditor
from .jr_delivery_script import as_srt, cue, local_path, parse_srt, read_book, sheet_json, validate


Cues = io.Custom("JR_DRAMA3_CUES")
Timeline = io.Custom("JR_DRAMA3_TIMELINE")
CATEGORY = "audio/JR Drama 3/编辑与交付"


def schema(cls, label, inputs, outputs, **options):
    return io.Schema(
        node_id=cls.__name__,
        display_name="JR Drama3 · " + label,
        category=CATEGORY,
        inputs=inputs,
        outputs=outputs,
        **options,
    )


def sheet_outputs():
    return [
        Cues.Output("cues"),
        io.String.Output("lines", is_output_list=True),
        io.Int.Output("indices", is_output_list=True),
        io.String.Output("cue_json"),
        io.String.Output("roles", is_output_list=True),
    ]


def sheet_result(rows):
    return io.NodeOutput(
        rows,
        [r["text"] for r in rows],
        list(range(1, len(rows) + 1)),
        sheet_json(rows),
        [r["role"] for r in rows],
    )


def source_fingerprint(root, name):
    if not name:
        return ""
    stat = local_path(root, name).stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


class JRDrama3ReadSRT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Read SRT / 导入字幕",
            [
                io.String.Input(
                    "srt", multiline=True, default="1\n00:00:00,000 --> 00:00:03,000\n【旁白】你好。"
                ),
                io.String.Input(
                    "input_file",
                    default="",
                    tooltip="Optional path relative to ComfyUI/input, overrides text.",
                ),
            ],
            sheet_outputs(),
        )

    @classmethod
    def fingerprint_inputs(cls, srt, input_file):
        return source_fingerprint(folder_paths.get_input_directory(), input_file)

    @classmethod
    def execute(cls, srt, input_file):
        if input_file:
            path = local_path(folder_paths.get_input_directory(), input_file)
            if path.suffix.lower() != ".srt" or path.stat().st_size > 64 * 1024**2:
                raise ValueError("Choose a UTF-8 SRT file under 64 MiB.")
            srt = path.read_text(encoding="utf-8-sig")
        return sheet_result(parse_srt(srt))


class JRDrama3ReadBook(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Read book / 导入书稿",
            [
                io.String.Input(
                    "input_file", default="book.txt", tooltip="UTF-8 TXT or EPUB, relative to ComfyUI/input."
                )
            ],
            sheet_outputs(),
        )

    @classmethod
    def fingerprint_inputs(cls, input_file):
        return source_fingerprint(folder_paths.get_input_directory(), input_file)

    @classmethod
    def execute(cls, input_file):
        return sheet_result(read_book(local_path(folder_paths.get_input_directory(), input_file)))


class JRDrama3CueSheet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Edit cue sheet / 编辑台词表",
            [
                io.String.Input(
                    "cue_json",
                    multiline=True,
                    default=sheet_json([cue(1, "Hello.", "Narrator", "Chapter 1")]),
                )
            ],
            sheet_outputs(),
        )

    @classmethod
    def execute(cls, cue_json):
        return sheet_result(validate(json.loads(cue_json)))


class JRDrama3CollectTakes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Collect generated lines / 收集逐句音频",
            [
                Cues.Input("cues"),
                io.Audio.Input("audio"),
                io.String.Input("settings_json", default="{}", multiline=True),
            ],
            [Timeline.Output("timeline")],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, cues, audio, settings_json):
        if len(cues) != 1:
            raise ValueError("Connect one cue sheet to Collect and connect the mapped per-line AUDIO output.")
        rows = validate(cues[0])
        if len(rows) != len(audio):
            raise ValueError(f"Expected one AUDIO per cue ({len(rows)}), received {len(audio)}.")
        if len(settings_json) not in (1, len(rows)):
            raise ValueError("Supply one settings JSON or one per take.")
        settings = [json.loads(s) for s in settings_json]
        if len(settings) == 1 and isinstance(settings[0], list):
            settings = settings[0]
            if len(settings) != len(rows) or any(
                not isinstance(s, dict) or str(s.get("cue")) != row["id"] for s, row in zip(settings, rows)
            ):
                raise ValueError("The batch report must match the cue IDs and order of the sheet.")
        if not all(isinstance(s, dict) for s in settings):
            raise ValueError("Take settings must be JSON objects.")
        return io.NodeOutput(
            [
                dict(cue=row, audio=a, settings=settings[i % len(settings)])
                for i, (row, a) in enumerate(zip(rows, audio, strict=True))
            ]
        )


class JRDrama3PutTake(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Add or replace take / 加入或替换录音",
            [
                Cues.Input("cues"),
                io.Audio.Input("audio"),
                io.Int.Input("index", default=1, min=1),
                io.String.Input("settings_json", multiline=True, default="{}"),
                Timeline.Input("timeline", optional=True),
            ],
            [Timeline.Output("timeline")],
        )

    @classmethod
    def execute(cls, cues, audio, index, settings_json, timeline=None):
        rows = validate(cues)
        if not 1 <= index <= len(rows):
            raise ValueError("Cue index is outside the sheet.")
        chosen = rows[index - 1]
        settings = json.loads(settings_json)
        if not isinstance(settings, dict):
            raise ValueError("Take settings must be a JSON object.")
        existing = {item["cue"]["id"]: item for item in timeline or []}
        if existing.keys() - {r["id"] for r in rows}:
            raise ValueError("Existing takes refer to cues missing from this sheet.")
        existing[chosen["id"]] = dict(cue=chosen, audio=audio, settings=settings)
        return io.NodeOutput([dict(existing[row["id"]], cue=row) for row in rows if row["id"] in existing])


def timing_controls():
    return [
        io.Int.Input("rate", default=44100, min=8000, max=192000),
        io.Combo.Input("overflow", options=["error", "extend", "time_stretch"]),
        io.Float.Input("max_speed", default=1.25, min=1, max=4, step=0.01),
        io.Combo.Input("overlap", options=["mix", "error"]),
        io.Int.Input("gap_ms", default=0, min=0, max=60000, tooltip="Gap between untimed lines in ms."),
    ]


class JRDrama3MixTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Mix timeline or stem / 混音或单角色分轨",
            [
                Timeline.Input("timeline"),
                *timing_controls(),
                io.String.Input(
                    "role_filter",
                    default="",
                    tooltip="Empty: full mix. Exact role or [unassigned]: isolated stem.",
                ),
            ],
            [io.Audio.Output("audio"), io.String.Output("srt"), io.String.Output("timing_json")],
        )

    @classmethod
    def execute(cls, timeline, rate, overflow, max_speed, overlap, gap_ms, role_filter):
        p = arrange(timeline, rate, overflow, max_speed, overlap, gap_ms)
        if max(i.slot_end for i in p) * max(i.x.shape[1] for i in p) * 4 > 1024**3:
            raise ValueError("In-memory preview exceeds 1 GiB. Use Export Project for streaming mixdown.")
        role = None if not role_filter else "" if role_filter == "[unassigned]" else role_filter
        rows = placed_rows(p, rate)
        if role is not None:
            rows = [r for r in rows if r["role"] == role]
            if not rows:
                raise ValueError("The role is absent from this timeline.")
        mixed = np.concatenate(list(blocks(p, role)), axis=0)
        mixed *= mix_gain(p)
        return io.NodeOutput(as_audio(mixed, rate), as_srt(rows), sheet_json(rows))


class JRDrama3ExportProject(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Export project & stems / 导出工程及分轨",
            [
                Timeline.Input("timeline"),
                io.String.Input("title", default="Drama3"),
                io.Combo.Input("primary_language", options=["primary", "zh", "en"]),
                *timing_controls(),
                io.Boolean.Input("stems", default=True),
                io.Boolean.Input("lines", default=True),
                io.Boolean.Input("m4b", default=False),
                io.Combo.Input("extra_format", options=["none", "mp3", "flac"]),
                io.String.Input("project_metadata_json", multiline=True, default="{}"),
                Timeline.Input("other_language", optional=True),
                io.Custom("DRAMA3_CAST").Input("cast", optional=True),
                io.Image.Input("cover", optional=True, tooltip="有声书封面。每种语言的 M4B 均嵌入此图。"),
                io.Boolean.Input(
                    "include_workflow",
                    default=False,
                    advanced=True,
                    tooltip="附带完整画布工作流及其组件设置。",
                ),
            ],
            [io.String.Output("project_path"), io.String.Output("manifest_json")],
            is_output_node=True,
            not_idempotent=True,
            hidden=[io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(
        cls,
        timeline,
        title,
        primary_language,
        rate,
        overflow,
        max_speed,
        overlap,
        gap_ms,
        stems,
        lines,
        m4b,
        extra_format,
        project_metadata_json,
        other_language=None,
        cast=None,
        include_workflow=False,
        cover=None,
    ):
        if other_language is not None and primary_language == "primary":
            raise ValueError("Select zh or en as the first language for bilingual delivery.")
        tags = [primary_language, "en" if primary_language == "zh" else "zh"]
        settings = dict(
            rate=rate,
            overflow=overflow,
            max_speed=max_speed,
            overlap=overlap,
            gap_ms=gap_ms,
            stems=stems,
            lines=lines,
            m4b=m4b,
            extra_format=extra_format,
        )
        workflow = None
        if include_workflow:
            workflow = (cls.hidden.extra_pnginfo or {}).get("workflow")
            if not isinstance(workflow, dict):
                raise ValueError(
                    "The request did not include a canvas workflow. Queue from the canvas or supply extra_pnginfo.workflow."
                )
        path, doc = export_project(
            folder_paths.get_output_directory(),
            title,
            timeline,
            other_language,
            tags,
            settings,
            json.loads(project_metadata_json),
            cast,
            workflow,
            cover,
        )
        return io.NodeOutput(path, sheet_json(doc), ui={"text": [path]})


class JRDrama3RestoreProject(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Restore project / 恢复工程",
            [
                io.String.Input("project_path", default=""),
                io.Combo.Input("location", options=["output", "input"]),
                io.Combo.Input("version", options=["first", "second"]),
            ],
            [
                Timeline.Output("timeline"),
                Cues.Output("cues"),
                io.String.Output("cue_json"),
                io.String.Output("manifest_json"),
                io.Audio.Output("takes", is_output_list=True),
                io.Custom("DRAMA3_CAST").Output("cast"),
                io.String.Output("workflow_json"),
                io.Image.Output("cover"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, project_path, location, version):
        root = (
            folder_paths.get_output_directory()
            if location == "output"
            else folder_paths.get_input_directory()
        )
        return source_fingerprint(root, project_path)

    @classmethod
    def execute(cls, project_path, location, version):
        root = (
            folder_paths.get_output_directory()
            if location == "output"
            else folder_paths.get_input_directory()
        )
        timeline, doc, cast, workflow, cover = restore_project(
            root, project_path, 0 if version == "first" else 1
        )
        rows = [item["cue"] for item in timeline]
        return io.NodeOutput(
            timeline,
            rows,
            sheet_json(rows),
            sheet_json(doc),
            [item["audio"] for item in timeline],
            cast,
            workflow,
            cover,
        )


class JRDrama3MasterAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Edit & master audio / 音频母带",
            [
                io.Audio.Input("audio"),
                io.Float.Input("start_s", default=0, min=0, max=86400),
                io.Float.Input("end_s", default=0, min=0, max=86400),
                io.Float.Input(
                    "trim_db", default=-120, min=-120, max=-12, tooltip="-120 disables silence trimming."
                ),
                io.Float.Input("fade_in_s", default=0, min=0, max=60),
                io.Float.Input("fade_out_s", default=0, min=0, max=60),
                io.Float.Input("gain_db", default=0, min=-60, max=24),
                io.Combo.Input("loudness", options=["off", "-16 LUFS", "-18 LUFS", "-20 LUFS", "-23 LUFS"]),
                io.Float.Input("peak_db", default=-1, min=-9, max=0),
                io.Combo.Input(
                    "resample_to", options=["source", "16000", "22050", "24000", "44100", "48000"]
                ),
            ],
            [io.Audio.Output("audio")],
        )

    @classmethod
    def execute(
        cls, audio, start_s, end_s, trim_db, fade_in_s, fade_out_s, gain_db, loudness, peak_db, resample_to
    ):
        return io.NodeOutput(
            postprocess(
                audio,
                start_s,
                end_s,
                trim_db,
                fade_in_s,
                fade_out_s,
                gain_db,
                0 if loudness == "off" else int(loudness.split()[0]),
                peak_db,
                0 if resample_to == "source" else int(resample_to),
            )
        )


class JRDrama3JoinAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Join audio / 拼接音频",
            [
                io.Audio.Input("first"),
                io.Audio.Input("second"),
                io.Float.Input("gap_s", default=0, min=0, max=60),
                io.Float.Input("crossfade_s", default=0, min=0, max=60),
            ],
            [io.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, first, second, gap_s, crossfade_s):
        if gap_s and crossfade_s:
            raise ValueError("Choose a gap or crossfade for the same join.")
        a, rate = samples(first)
        b, rate2 = samples(second)
        b = resample(b, rate2, rate)
        channels = max(a.shape[1], b.shape[1])
        a = np.broadcast_to(a, (len(a), channels))
        b = np.broadcast_to(b, (len(b), channels))
        n = round(rate * crossfade_s)
        if n > min(len(a), len(b)):
            raise ValueError("Crossfade exceeds the length of one clip.")
        if n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
            parts = [a[:-n], a[-n:] * (1 - ramp) + b[:n] * ramp, b[n:]]
        else:
            parts = [a, np.zeros((round(rate * gap_s), channels), np.float32), b]
        return io.NodeOutput(as_audio(np.concatenate(parts), rate))


NODES = [
    JRDrama3ReadSRT,
    JRDrama3ReadBook,
    JRDrama3CueSheet,
    JRDrama3CollectTakes,
    JRDrama3PutTake,
    JRDrama3MixTimeline,
    JRDrama3TimelineEditor,
    JRDrama3ExportProject,
    JRDrama3RestoreProject,
    JRDrama3MasterAudio,
    JRDrama3JoinAudio,
]
