"""Optional provisioning; network access is used only by explicitly running this script."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
from uuid import uuid4

AUX = Path(__file__).resolve().parent / ".runtime" / "aux"
WHISPER_SHA = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
MODEL_REVISIONS = {
    "opus-mt-en-zh": "408d9bc410a388e1d9aef112a2daba955b945255",
    "opus-mt-zh-en": "cf109095479db38d6df799875e34039d4938aaa6",
}
OPUS_FILES = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
]


def digest(path, size, expected):
    h = hashlib.sha256() if len(expected) == 64 else hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        while block := f.read(2 << 20):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, size, checksum):
    if path.is_file() and path.stat().st_size == size and digest(path, size, checksum) == checksum:
        return
    chunk = 4 << 20
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging prevents two explicit setup processes publishing each other's partial files.
    staging = path.with_name(path.name + "." + uuid4().hex + ".part")
    count = (size + chunk - 1) // chunk

    def read(i):
        start = i * chunk
        end = min(size, start + chunk) - 1
        headers = {"Accept-Encoding": "identity"}
        if count > 1:
            headers["Range"] = f"bytes={start}-{end}"
        for attempt in range(5):
            try:
                req = Request(url + f"?download=true&offset={start}&try={attempt}", headers=headers)
                with urlopen(req, timeout=35) as resp:
                    if count > 1 and resp.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise IOError("Server did not honor this range")
                    b = resp.read(end - start + 2)
                if len(b) != end - start + 1:
                    raise IOError("Incomplete response")
                return b
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)

    try:
        with staging.open("wb") as out, ThreadPoolExecutor(max_workers=5) as pool:
            # Bounded batches avoid retaining a file's worth of completed futures in RAM.
            for i in range(0, count, 8):
                for block in pool.map(read, range(i, min(i + 8, count))):
                    out.write(block)
        if digest(staging, size, checksum) != checksum:
            raise ValueError(f"Checksum mismatch: {path.name}")
        os.replace(staging, path)
        print(f"Verified {path.name}: {checksum}", flush=True)
    finally:
        staging.unlink(missing_ok=True)


def opus(name, revision, out):
    base = f"https://huggingface.co/Helsinki-NLP/{name}"
    endpoint = f"https://huggingface.co/api/models/Helsinki-NLP/{name}/revision/{revision}?blobs=true"
    with urlopen(endpoint, timeout=30) as resp:
        meta = json.load(resp)
    if meta["sha"] != revision:
        raise ValueError("Unexpected OPUS model revision.")
    files = {s["rfilename"]: s for s in meta["siblings"]}
    for f in OPUS_FILES:
        info = files[f]
        key = (info.get("lfs") or {}).get("sha256") or info["blobId"]
        fetch(f"{base}/resolve/{revision}/{f}", out / name / f, info["size"], key)


def main():
    parser = argparse.ArgumentParser(description="Download and verify optional local Drama 3 analysis models")
    parser.add_argument("models", nargs="+", choices=["whisper", "en-zh", "zh-en"])
    parser.add_argument("--output-dir", type=Path, default=AUX)
    a = parser.parse_args()
    for m in dict.fromkeys(a.models):
        if m == "whisper":
            url = f"https://openaipublic.azureedge.net/main/whisper/models/{WHISPER_SHA}/small.pt"
            fetch(url, a.output_dir / "whisper-small.pt", 483617219, WHISPER_SHA)
        else:
            name = "opus-mt-" + m
            opus(name, MODEL_REVISIONS[name], a.output_dir)


if __name__ == "__main__":
    main()
