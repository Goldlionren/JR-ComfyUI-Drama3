"""On-disk voice versions shared by the ComfyUI nodes and the local speech API."""

from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np


def library_path(default):
    value = os.getenv("JR_DRAMA3_VOICE_DB")
    if value and ("://" in value or value.startswith(("//", "\\\\"))):
        raise ValueError("音色库必须位于本机磁盘。")
    return Path(value).resolve() if value else Path(default)


def save_voice(db, name, language, description, tags, transcript, waveform, rate):
    name, transcript = name.strip(), transcript.strip()
    if not name or not transcript:
        raise ValueError("A saved voice needs a name and the exact transcript of its reference")
    raw = io.BytesIO()
    np.save(raw, np.ascontiguousarray(waveform), allow_pickle=False)
    wave_bytes = raw.getvalue()
    ref = {"name": name, "language": language.strip(), "description": description.strip(),
           "tags": tags.strip(), "transcript": transcript, "sample_rate": rate, "seconds": waveform.shape[-1] / rate}
    # The field order, JSON formatting and NPY encoding are part of existing ComfyUI version IDs.
    digest = hashlib.sha256(json.dumps(ref, ensure_ascii=False, indent=2).encode() + b"\x00" + wave_bytes).hexdigest()
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, timeout=30)) as con, con:
        con.execute("""CREATE TABLE IF NOT EXISTS voices (
            row_id INTEGER PRIMARY KEY, voice_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL, language TEXT NOT NULL, description TEXT NOT NULL, tags TEXT NOT NULL,
            transcript TEXT NOT NULL, sample_rate INTEGER NOT NULL, seconds REAL NOT NULL, samples BLOB NOT NULL)""")
        con.execute("""INSERT OR IGNORE INTO voices
            (voice_id,name,language,description,tags,transcript,sample_rate,seconds,samples)
            VALUES(?,?,?,?,?,?,?,?,?)""",
                    (digest, name, ref["language"], ref["description"], ref["tags"], transcript,
                     rate, ref["seconds"], zlib.compress(wave_bytes, level=1)))
    return digest


def load_voice(db, voice_id):
    path = Path(db)
    if not path.is_file():
        raise ValueError("The local voice library is empty")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM voices WHERE voice_id=?", (voice_id,)).fetchone()
    if not row:
        raise ValueError(f"No local voice with ID {voice_id!r}")
    item = dict(row)
    samples = np.load(io.BytesIO(zlib.decompress(item.pop("samples"))), allow_pickle=False)
    return samples, item
