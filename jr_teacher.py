"""Offline verification of the complete October 29 Fish Audio teacher."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "resources" / "teacher.json").read_text(encoding="utf-8"))
REVISION = MANIFEST["revision"]


class FlaTeacherPath(str):
    """A teacher path carrying the node's FLA selection through existing path consumers."""


def _local_path(value):
    raw = str(value).strip()
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise ValueError("教师权重必须位于本机磁盘。")
    return Path(raw).expanduser().resolve()


def locate_teacher(directory=""):
    """Comfy model folders support its extra_model_paths configuration."""
    if directory.strip():
        return _local_path(directory)
    configured = os.environ.get("JR_DRAMA3_MODEL_DIR")
    if configured:
        return _local_path(configured)
    from folder_paths import get_folder_paths, models_dir
    paths = [ROOT / ".runtime" / "teacher"]
    try:
        paths.extend(Path(p) for p in get_folder_paths("drama3"))
    except KeyError:
        pass
    paths.extend([Path(models_dir) / "drama3", Path(models_dir) / "tts" / "drama-3"])
    return next((p for p in paths if (p / "config.json").is_file()), paths[0])


def _stamp(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino


@lru_cache(maxsize=128)
def _verify_file(name, stamp, expected):
    path = Path(name)
    size = stamp[0]
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if _stamp(path) != stamp or digest.hexdigest() != expected:
        raise ValueError(f"校验失败，请恢复官方 78e5ab7 快照中的文件：{path.name}")


def validate_teacher(directory):
    path = _local_path(directory)
    if list(path.glob("model-*-of-00008.safetensors")):
        raise ValueError("检测到早期八分片。请使用单独目录保存官方新版 78e5ab7 十五分片教师。")
    files = MANIFEST["files"]
    missing = [name for name in files if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} 缺少新版教师文件：{', '.join(missing)}")
    for name, spec in files.items():
        file = path / name
        stamp = _stamp(file)
        if stamp[0] != spec["bytes"]:
            raise ValueError(f"{name} 大小与新版官方快照不符：{stamp[0]} / {spec['bytes']}")
        _verify_file(str(file), stamp, spec.get("sha256") or spec["git_blob"])
    return path
