"""Portable project and audio-book exports. All public paths are relative."""

import json
import io
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import soundfile as sf
import numpy as np
import torch
from PIL import Image

from .jr_delivery_audio import arrange, as_audio, convert, mix_gain, placed_rows, samples, save_mix
from .jr_delivery_script import as_srt, local_path, sheet_json, validate


def short_name(name):
    name = str(name).strip()
    if not re.fullmatch(r"[\w -]{1,64}", name, re.UNICODE):
        raise ValueError("Use a title of 1–64 letters, digits, spaces, hyphens or underscores.")
    return name


def pair_check(primary, second):
    first = [(x["cue"]["id"], x["cue"]["role"]) for x in primary]
    other = [(x["cue"]["id"], x["cue"]["role"]) for x in second]
    if first != other:
        raise ValueError("Bilingual takes must share the same line IDs, roles and order.")


def ffmeta(value):
    return "".join("\\" + ch if ch in "\\=;#\n" else ch for ch in str(value))


def save_cover(path, cover):
    if (
        not isinstance(cover, torch.Tensor)
        or cover.ndim != 4
        or cover.shape[0] != 1
        or cover.shape[-1] not in (3, 4)
    ):
        raise ValueError("Cover requires one RGB or RGBA ComfyUI IMAGE.")
    image = cover[0].detach().float().cpu().numpy()
    if not image.size or not np.isfinite(image).all():
        raise ValueError("Invalid cover image pixels.")
    data = Image.fromarray(np.rint(image.clip(0, 1) * 255).astype("uint8"))
    data.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
    data.save(path, format="PNG")


def save_m4b(path, wave_path, placements, rate, title, cover_path=None):
    sections = []
    for p in sorted(placements, key=lambda p: p.start):
        name = p.cue["chapter"] or title
        if not sections or name != sections[-1][1]:
            if sections and p.start == sections[-1][0]:
                raise ValueError("Two chapters start at the same time.")
            sections.append((p.start if sections else 0, name))
    end = max(p.slot_end for p in placements)
    metadata = [";FFMETADATA1", "title=" + ffmeta(title)]
    for i, (start, name) in enumerate(sections):
        stop = sections[i + 1][0] if i + 1 < len(sections) else end
        metadata.extend(
            ("[CHAPTER]", f"TIMEBASE=1/{rate}", f"START={start}", f"END={stop}", "title=" + ffmeta(name))
        )
    meta_path = path.with_suffix(".ffmeta")
    meta_path.write_text("\n".join(metadata) + "\n", encoding="utf-8")
    try:
        convert(
            [
                "-i",
                wave_path,
                "-f",
                "ffmetadata",
                "-i",
                meta_path,
                *(["-i", cover_path] if cover_path else []),
                "-map",
                "0:a:0",
                *(
                    ["-map", "2:v:0", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
                    if cover_path
                    else []
                ),
                "-map_metadata",
                "1",
                "-map_chapters",
                "1",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-f",
                "mp4",
                "-movflags",
                "+faststart",
                path,
            ]
        )
    finally:
        meta_path.unlink(missing_ok=True)


def save_language(stage, z, tag, timeline, title, settings, cover_path=None):
    placements = arrange(
        timeline, **{k: settings[k] for k in ("rate", "overflow", "max_speed", "overlap", "gap_ms")}
    )
    rate = settings["rate"]
    rows = placed_rows(placements, rate)
    n_stems = len({p.cue["role"] for p in placements}) if settings["stems"] else 0
    frame_bytes = max(p.x.shape[1] for p in placements) * 4
    required = (n_stems + 1) * max(p.slot_end for p in placements) * frame_bytes
    required += sum(item["audio"]["waveform"].numel() * 4 for item in timeline)
    if settings["lines"]:
        required += sum(p.x.nbytes for p in placements)
    if required > shutil.disk_usage(stage).free:
        raise ValueError(
            f"Insufficient disk space for this delivery (at least {required / 1024**3:.2f} GiB). Check the timecodes and stem count."
        )
    gain = mix_gain(placements)
    destination = stage / tag
    destination.mkdir()
    master = destination / "mix.wav"
    save_mix(master, placements, rate, gain)
    (destination / "captions.srt").write_text(as_srt(rows), encoding="utf-8")
    manifest = []
    for index, (item, row, p) in enumerate(zip(timeline, rows, placements, strict=True), 1):
        x, source_rate = samples(item["audio"])
        source = f"{tag}/source/{index:06}.wav"
        with tempfile.NamedTemporaryFile(dir=stage, suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(tmp_path, x, source_rate, format="WAV", subtype="FLOAT")
            z.write(tmp_path, source)
        finally:
            tmp_path.unlink(missing_ok=True)
        part = dict(
            cue=item["cue"],
            placed=row,
            audio_start_sample=p.start,
            audio_end_sample=p.end,
            slot_end_sample=p.slot_end,
            source=source,
            source_sample_rate=source_rate,
            source_frames=len(x),
            settings=item.get("settings", {}),
            edit=item.get("edit", {}),
        )
        if settings["lines"]:
            out = destination / "lines" / f"{index:06}.wav"
            out.parent.mkdir(exist_ok=True)
            sf.write(out, p.x * gain, rate, subtype="FLOAT")
            part["line_file"] = out.relative_to(stage).as_posix()
        manifest.append(part)
    tracks = []
    if settings["stems"]:
        for index, role in enumerate(dict.fromkeys(p.cue["role"] for p in placements), 1):
            path = destination / "stems" / f"{index:03}.wav"
            path.parent.mkdir(exist_ok=True)
            save_mix(path, placements, rate, gain, role=role)
            tracks.append(dict(role=role, file=path.relative_to(stage).as_posix()))
    if settings["m4b"]:
        save_m4b(destination / "book.m4b", master, placements, rate, title, cover_path)
    if settings["extra_format"] in ("flac", "mp3"):
        fmt = settings["extra_format"]
        convert(
            ["-i", master, "-c:a", {"flac": "flac", "mp3": "libmp3lame"}[fmt], destination / ("mix." + fmt)]
        )
    return dict(lines=manifest, stems=tracks, gain=gain)


def export_project(
    root, name, timeline, second, tags, settings, metadata, cast=None, workflow=None, cover=None
):
    title = short_name(name)
    if second is not None:
        pair_check(timeline, second)
    if not isinstance(metadata, dict):
        raise ValueError("Project metadata must be a JSON object.")
    folder = local_path(root, "JR-Drama3/projects")
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / (title + "-" + uuid.uuid4().hex[:12])
    with tempfile.TemporaryDirectory(dir=folder, prefix=".prepare-") as tmp:
        stage = Path(tmp) / "project"
        stage.mkdir()
        with ZipFile(stage / "project.jrdrama.zip", "w", compression=ZIP_STORED, allowZip64=True) as z:
            cover_path = None
            if cover is not None:
                cover_path = stage / "cover.png"
                save_cover(cover_path, cover)
                z.write(cover_path, "cover.png")
            versions = {}
            for tag, track in zip(tags, [timeline, second]):
                if track is not None:
                    versions[tag] = save_language(stage, z, tag, track, title, settings, cover_path)
            voices = []
            for i, item in enumerate(cast or []):
                voice = {k: v for k, v in item.items() if k != "audio"}
                if item.get("audio") is not None:
                    x, rate = samples(item["audio"])
                    member = f"cast/{i:04}.wav"
                    data = io.BytesIO()
                    sf.write(data, x, rate, subtype="FLOAT", format="WAV")
                    z.writestr(member, data.getbuffer())
                    voice["reference_wav"] = member
                voices.append(voice)
            if workflow is not None:
                workflow_text = sheet_json(workflow)
                z.writestr("workflow.json", workflow_text.encode("utf-8"))
                (stage / "workflow.json").write_text(workflow_text, encoding="utf-8")
            document = dict(
                format="JRDrama3.project",
                version=1,
                title=title,
                settings=settings,
                metadata=metadata,
                languages=versions,
                cast=voices,
                workflow="workflow.json" if workflow is not None else None,
                cover="cover.png" if cover_path else None,
            )
            text = sheet_json(document)
            z.writestr("manifest.json", text.encode("utf-8"))
        (stage / "manifest.json").write_text(text, encoding="utf-8")
        os.replace(stage, final)
    return (final / "project.jrdrama.zip").relative_to(Path(root).resolve()).as_posix(), document


def restore_project(root, name, which):
    path = local_path(root, name)
    if not path.name.endswith(".jrdrama.zip"):
        raise ValueError("Choose a .jrdrama.zip project.")
    with ZipFile(path) as z:
        if z.getinfo("manifest.json").file_size > 64 * 1024**2:
            raise ValueError("Project manifest exceeds 64 MiB.")
        doc = json.loads(z.read("manifest.json"))
        if doc.get("format") != "JRDrama3.project" or doc.get("version") != 1:
            raise ValueError("Unsupported Drama3 project version.")
        languages = list(doc["languages"])
        if which < 0 or which >= len(languages):
            raise ValueError("The selected language track is absent from this project.")
        tag = languages[which]
        if tag not in ("primary", "zh", "en"):
            raise ValueError("Unsupported project language key.")
        lines = doc["languages"][tag]["lines"]
        rows = validate([r["cue"] for r in lines])
        result = []
        for i, (row, line) in enumerate(zip(rows, lines, strict=True), 1):
            member = f"{tag}/source/{i:06}.wav"
            if line["source"] != member:
                raise ValueError(f"Invalid source path for cue {row['id']}.")
            info = z.getinfo(member)
            if info.compress_type != ZIP_STORED:
                raise ValueError("Project audio must use the uncompressed WAV archive format.")
            with z.open(info) as stream, sf.SoundFile(stream) as f:
                if (
                    f.channels not in (1, 2)
                    or f.format != "WAV"
                    or f.subtype != "FLOAT"
                    or not (0 < f.frames <= info.file_size // (4 * f.channels))
                ):
                    raise ValueError(f"Invalid archived WAV for cue {row['id']}.")
                if (f.frames, f.samplerate) != (line["source_frames"], line["source_sample_rate"]):
                    raise ValueError(f"Source duration differs from manifest at cue {row['id']}.")
                a = as_audio(f.read(dtype="float32", always_2d=True), f.samplerate)
            result.append(
                dict(cue=row, audio=a, settings=line.get("settings", {}), edit=line.get("edit", {}))
            )
        cast = []
        for i, original in enumerate(doc.get("cast", [])):
            voice = dict(original)
            ref = voice.pop("reference_wav", None)
            if ref is not None:
                member = f"cast/{i:04}.wav"
                if ref != member or z.getinfo(member).compress_type != ZIP_STORED:
                    raise ValueError("Invalid archived cast reference.")
                with z.open(member) as stream, sf.SoundFile(stream) as f:
                    if (
                        f.channels not in (1, 2)
                        or f.format != "WAV"
                        or f.subtype != "FLOAT"
                        or not (0 < f.frames <= z.getinfo(member).file_size // (4 * f.channels))
                    ):
                        raise ValueError("Invalid archived cast WAV.")
                    voice["audio"] = as_audio(f.read(dtype="float32", always_2d=True), f.samplerate)
            else:
                voice["audio"] = None
            cast.append(voice)
        workflow = ""
        if doc.get("workflow"):
            if doc["workflow"] != "workflow.json" or z.getinfo("workflow.json").file_size > 64 * 1024**2:
                raise ValueError("Invalid archived workflow.")
            workflow = z.read("workflow.json").decode("utf-8")
            if not isinstance(json.loads(workflow), dict):
                raise ValueError("Archived workflow must contain a JSON object.")
        cover = None
        if doc.get("cover"):
            if doc["cover"] != "cover.png" or z.getinfo("cover.png").file_size > 48 * 1024**2:
                raise ValueError("Invalid project cover.")
            with z.open("cover.png") as stream, Image.open(stream, formats=["PNG"]) as image:
                if max(image.size) > 3000 or image.mode not in ("RGB", "RGBA"):
                    raise ValueError("Unsupported project cover format.")
                cover = torch.from_numpy(np.array(image, dtype=np.float32) / 255)[None]
    return result, doc, cast, workflow, cover
