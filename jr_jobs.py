"""Durable, content-addressed takes; files never contain pickle objects."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file


class IncompleteTakeError(ValueError):
    """A sampled take cannot be delivered, locked or used as history."""


def is_complete(report):
    reason = report.get("finish_reason")
    return reason == "eos" or (reason == "empty_infill" and report.get("mode") == "infill")


def require_complete(report):
    if not is_complete(report):
        raise IncompleteTakeError(
            f"台词未完整结束（{report.get('finish_reason', 'unknown')}）。"
            "请提高 max_frames、缩短台词或换一个种子。"
        )


def as_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def key_for(value):
    return hashlib.sha256(as_json(value).encode("utf-8")).hexdigest()


def cast_key(cast):
    roles = []
    for ref in cast or ():
        meta = {k: ref.get(k) for k in ("speaker", "name", "transcript", "description")}
        audio = ref.get("audio")
        if audio is not None:
            wave = audio["waveform"].detach().float().cpu().contiguous().numpy()
            meta["audio"] = [int(audio["sample_rate"]), list(wave.shape), hashlib.sha256(wave).hexdigest()]
        roles.append(meta)
    return key_for(roles)


def read_take(folder, key):
    path = Path(folder) / (key + ".safetensors")
    if not path.is_file():
        return None
    with safe_open(path, framework="pt", device="cpu") as handle:
        meta = json.loads(handle.metadata()["take"])
        if meta["key"] != key:
            raise ValueError("候选文件的签名与名称不一致。")
        if not is_complete(meta["report"]):
            # Older releases could commit truncated runs. Leave the file
            # alone until a complete replacement has been generated.
            return None
        wave = handle.get_tensor("waveform")
    return ({"waveform": wave, "sample_rate": meta["sample_rate"]}, meta["history"], meta["report"])


def save_take(folder, key, result):
    audio, history, report = result
    require_complete(report)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = as_json({"key": key, "sample_rate": audio["sample_rate"], "history": history, "report": report})
    fd, staging = tempfile.mkstemp(prefix=key + "-", suffix=".part", dir=folder)
    os.close(fd)
    try:
        save_file({"waveform": audio["waveform"].detach().cpu().contiguous()}, staging,
                  metadata={"take": meta})
        try:
            os.replace(staging, folder / (key + ".safetensors"))
        except PermissionError:
            # Windows may forbid replacing a take held by an active reader.
            # A completed concurrent writer of the same input satisfies this save.
            if read_take(folder, key) is None:
                raise
    finally:
        Path(staging).unlink(missing_ok=True)
