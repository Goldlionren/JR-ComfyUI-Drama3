"""Versioned, local voice references. One SQLite transaction commits both audio and metadata."""

from __future__ import annotations

import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import torch

from .jr_script_core import checked_cast, normalize_name
from .jr_voice_store import library_path, load_voice, save_voice


def prepare_reference(audio, start_seconds=0, end_seconds=0, trim_edges=False, silence_db=-50,
                      normalize_peak=False):
    wave, rate = audio["waveform"], audio["sample_rate"]
    if not torch.is_tensor(wave) or wave.ndim != 3 or wave.shape[0] != 1 or min(wave.shape) == 0:
        raise ValueError("Expected one ComfyUI AUDIO reference with shape [1, channels, samples]")
    if type(rate) is not int or rate <= 0 or any(not math.isfinite(n) for n in (start_seconds, end_seconds, silence_db)):
        raise ValueError("Invalid sample rate, trim range or silence threshold")
    if start_seconds < 0 or end_seconds < 0:
        raise ValueError("Reference times must be nonnegative")
    first = round(start_seconds * rate)
    end = min(wave.shape[-1], round(end_seconds * rate)) if end_seconds else wave.shape[-1]
    if first >= end:
        raise ValueError("The reference selection is empty")
    clip = wave.detach().to(device="cpu", dtype=torch.float32)[:, :, first:end]
    if not bool(torch.isfinite(clip).all()):
        raise ValueError("The reference contains NaN or infinite samples")
    if trim_edges:
        above = torch.where(clip.abs().amax(dim=(0, 1)) > 10 ** (silence_db / 20))[0]
        if not above.numel():
            raise ValueError("The selected reference is silent at this threshold")
        left, right = int(above[0]), int(above[-1]) + 1
        clip = clip[:, :, left:right]
        first, end = first + left, first + right
    if normalize_peak:
        peak = float(clip.abs().max())
        if peak > 0:
            clip = clip * (10 ** (-1 / 20) / peak)
    return {"waveform": clip.contiguous(), "sample_rate": rate}, {"start_sample": first, "end_sample": end,
            "seconds": (end - first) / rate, "sample_rate": rate}


def voice_db():
    import folder_paths
    return library_path(Path(folder_paths.get_user_directory()) / "jr_drama3" / "voices.sqlite3")


def save_reference(db, name, language, description, tags, transcript, audio):
    clip, _ = prepare_reference(audio)
    return save_voice(db, name, language, description, tags, transcript, clip["waveform"].numpy(), clip["sample_rate"])


def _tokens(description):
    tokens = set(re.findall(r"[a-z0-9]+", description.casefold()))
    for phrase in re.findall(r"[\u3400-\u9fff]+", description):
        tokens.update(phrase)
        tokens.update(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return tokens


def find_references(db, query="", language="", distinct_names=False):
    if not Path(db).is_file():
        return []
    sql = "SELECT row_id,voice_id,name,language,description,tags,transcript,sample_rate,seconds FROM voices"
    with closing(sqlite3.connect(db, timeout=30)) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute(sql + " ORDER BY row_id DESC")]
    terms = _tokens(query)
    language = normalize_name(language)
    for row in rows:
        info = " ".join(str(row[k]) for k in ("name", "language", "description", "tags"))
        tokens = _tokens(info)
        exact = bool(query.strip()) and normalize_name(query) in (normalize_name(row["name"]), row["voice_id"])
        row["score"] = 100.0 if exact else len(terms & tokens) / max(1, len(terms))
    matches = sorted((x for x in rows if (not terms or x["score"] > 0)
                      and (not language or normalize_name(x["language"]) == language)),
                     key=lambda x: (x["score"], x["row_id"]), reverse=True)
    if distinct_names:
        seen = set()
        distinct = []
        for row in matches:
            name = normalize_name(row["name"])
            if name not in seen:
                distinct.append(row)
                seen.add(name)
        return distinct
    return matches


def get_reference(db, voice_id):
    samples, item = load_voice(db, voice_id)
    return {"waveform": torch.from_numpy(samples), "sample_rate": item["sample_rate"]}, item


def auto_cast(db, cast, overrides=None):
    cast = checked_cast(cast)
    if not cast:
        raise ValueError("Connect a cast from the Script or Voice Reference node")
    overrides = overrides or {}
    if not isinstance(overrides, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in overrides.items()):
        raise ValueError("Overrides must be a JSON object mapping role names to voice IDs")
    keys = {normalize_name(k): v for k, v in overrides.items()}
    unknown = set(keys) - {normalize_name(r["name"]) for r in cast}
    if unknown:
        raise ValueError(f"Unrecognized role overrides: {sorted(unknown)}")
    bound = {r["speaker"] for r in cast if r.get("audio") is not None}
    used = {normalize_name(r.get("voice_name") or r["name"]) for r in cast if r.get("audio") is not None}
    picked = {v: get_reference(db, v) for v in set(keys.values())}
    used.update(normalize_name(item["name"]) for _, item in picked.values())
    output, report, assigned_ids = [], [], set()
    for role in cast:
        key = normalize_name(role["name"])
        chosen = keys.get(key)
        if not chosen and role["speaker"] in bound:
            output.append(role)
            continue
        if role["speaker"] in assigned_ids:
            continue
        assigned_ids.add(role["speaker"])
        if not chosen:
            pool = find_references(db, role.get("description") or role["name"], role.get("language", ""),
                                   distinct_names=True)
            chosen = next((r["voice_id"] for r in pool if normalize_name(r["name"]) not in used), None)
        if not chosen:
            output.append(role)
            report.append({"role": role["name"], "selection": None, "reason": "No matching unused local voice"})
            continue
        audio, source = picked[chosen] if chosen in picked else get_reference(db, chosen)
        used.add(normalize_name(source["name"]))
        output.append({**role, "transcript": source["transcript"], "audio": audio,
                       "voice_id": chosen, "voice_name": source["name"]})
        report.append({"role": role["name"], "voice": source["name"], "voice_id": chosen,
                       "selection": "override" if key in keys else "description"})
    return output, report
