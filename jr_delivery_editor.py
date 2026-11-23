"""Non-destructive timeline edits and local graph audition assets."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import folder_paths
import numpy as np
import soundfile as sf
from comfy_api.latest import io

from .jr_delivery_audio import as_audio, samples
from .jr_delivery_script import local_path, sheet_json, validate


def audio_key(x, rate):
    digest = hashlib.sha256(f"jr-f32-v1/{rate}/{x.shape}".encode())
    digest.update(np.ascontiguousarray(x, dtype="<f4"))
    return digest.hexdigest()


def cache_audio(folder, key, x, rate, subtype):
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / (key + ".wav")
    if not path.exists():
        fd, temp_name = tempfile.mkstemp(dir=folder, suffix=".wav")
        os.close(fd)
        try:
            sf.write(temp_name, x, rate, subtype=subtype)
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
    return path


def waveform_envelope(x, bins=48):
    edges = np.linspace(0, len(x), min(bins, len(x)) + 1, dtype=np.int64)
    return [round(float(np.max(np.abs(x[a:b]))), 3) for a, b in zip(edges[:-1], edges[1:])]


def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer in milliseconds.")
    return value


class JRDrama3TimelineEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="Drama 3 · 分轨台本编辑与试听",
            category="audio/JR Drama 3/编辑与交付",
            is_output_node=True,
            description="运行一次载入录音，编辑后再次运行。裁切保留原始录音，角色与字幕修改同步到交付节点。",
            inputs=[
                io.Custom("JR_DRAMA3_TIMELINE").Input("timeline"),
                io.String.Input(
                    "edits_json",
                    default="{}",
                    advanced=True,
                    tooltip="画布编辑器自动保存的变更，可随工作流保存。",
                ),
            ],
            outputs=[
                io.Custom("JR_DRAMA3_TIMELINE").Output("timeline"),
                io.Custom("JR_DRAMA3_CUES").Output("cues"),
                io.String.Output("cue_json"),
            ],
        )

    @classmethod
    def execute(cls, timeline, edits_json="{}"):
        patches = json.loads(edits_json or "{}")
        rows = validate([item["cue"] for item in timeline])
        if not isinstance(patches, dict) or set(patches) - {r["id"] for r in rows}:
            raise ValueError(
                "Editor changes reference unknown cue IDs. Clear the editor or reconnect its source."
            )
        previews = local_path(folder_paths.get_temp_directory(), "jr_drama3_audition")
        locked_dir = local_path(folder_paths.get_output_directory(), "JR-Drama3/locked-takes")
        result, visible, cursor_ms = [], [], 0
        for original, row in zip(timeline, rows, strict=True):
            identifier = row["id"]
            patch = patches.get(identifier, {})
            if not isinstance(patch, dict):
                raise ValueError(f"Invalid edit for cue {identifier}.")
            audio = original["audio"]
            x, rate = samples(audio)
            key = audio_key(x, rate)
            edit = dict(original.get("edit", {}))
            bound = patch.get("source", key)
            locked = patch.get("locked", bool(edit.get("lock_source")))
            if (
                type(locked) is not bool
                or not isinstance(bound, str)
                or not re.fullmatch("[0-9a-f]{64}", bound)
            ):
                raise ValueError(f"Invalid take binding for cue {identifier}.")
            if locked:
                bound = patch.get("source") or edit.get("lock_source") or key
                if not isinstance(bound, str) or not re.fullmatch("[0-9a-f]{64}", bound):
                    raise ValueError("Invalid locked source.")
                if bound != key:
                    stored = locked_dir / (bound + ".wav")
                    if not stored.is_file():
                        raise ValueError(
                            f"Locked take for cue {identifier} is missing. Restore its project or unlock it."
                        )
                    x, rate = sf.read(stored, dtype="float32", always_2d=True)
                    if audio_key(x, rate) != bound:
                        raise ValueError(f"Stored take for cue {identifier} failed its content check.")
                    key, audio = bound, as_audio(x, rate)
                cache_audio(locked_dir, key, x, rate, "FLOAT")
                edit["lock_source"] = key
            else:
                if bound != key and any(k in patch for k in ("in_ms", "out_ms")):
                    raise ValueError(
                        f"Cue {identifier} has new audio. Clear or rebind old trim points first."
                    )
                edit.pop("lock_source", None)
            for field in ("text", "role", "chapter", "start_ms", "end_ms"):
                if field in patch:
                    row[field] = patch[field]
            row = validate([row])[0]
            raw_ms = max(1, round(len(x) * 1000 / rate))
            first = nonnegative(patch.get("in_ms", edit.get("in_ms", 0)), "Source in")
            last = patch.get("out_ms", edit.get("out_ms"))
            if last == raw_ms:
                last = None
            if last is not None:
                nonnegative(last, "Source out")
            if first >= (raw_ms if last is None else last) or (last is not None and last > raw_ms):
                raise ValueError(f"Source trim for cue {identifier} lies outside its recording.")
            edit.update(in_ms=first, out_ms=last)
            result.append(dict(original, cue=row, edit=edit, audio=audio))
            cache_audio(previews, key, x, rate, "PCM_16")
            length_ms = (last or raw_ms) - first
            start_ms = row["start_ms"] if row["start_ms"] is not None else cursor_ms
            slot_end = row["end_ms"] if row["end_ms"] is not None else start_ms + length_ms
            cursor_ms = max(cursor_ms, start_ms + length_ms, slot_end)
            base = dict(
                original["cue"],
                in_ms=original.get("edit", {}).get("in_ms", 0),
                out_ms=original.get("edit", {}).get("out_ms"),
                locked=bool(original.get("edit", {}).get("lock_source")),
            )
            visible.append(
                dict(
                    **row,
                    base=base,
                    key=key,
                    raw_ms=raw_ms,
                    in_ms=first,
                    out_ms=last,
                    locked=locked,
                    placed_start_ms=start_ms,
                    placed_end_ms=slot_end,
                    peaks=waveform_envelope(x),
                    preview=key + ".wav",
                )
            )
        new_rows = [item["cue"] for item in result]
        return io.NodeOutput(result, new_rows, sheet_json(new_rows), ui={"jr_drama3_timeline": visible})
