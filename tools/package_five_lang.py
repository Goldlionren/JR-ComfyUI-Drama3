"""Assemble the Windows inference ZIP from a prepared checkout."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jr_realtime_paths import MODELS, RUNTIME, engine_binary, model_file


def package(path):
    model = model_file("5lang")
    receipt = model.with_suffix(".gguf.sha256")
    with model.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != receipt.read_text("ascii").strip():
            raise ValueError("Five-language package checksum mismatch")
    exe = engine_binary("5lang")
    native = exe.relative_to(ROOT)
    cuda = Path(os.environ["CUDA_PATH"])
    entries = {}
    project = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT)
    for raw in project.split(b"\0"):
        if not raw:
            continue
        name = Path(raw.decode("utf-8"))
        if name.parts[0] in {"resources", "tools", "example_workflows", "licenses", "docs", "provider",
                            "jr_runtime", "web", "integrations"} or (
                len(name.parts) == 1 and (name.suffix == ".py" or name.name in {"README.md", "NOTICE.txt"})):
            entries[name] = ROOT / name
    dest_model = Path(".runtime") / MODELS["5lang"][0]
    entries.update({native: exe, dest_model: model, dest_model.with_suffix(".gguf.sha256"): receipt})
    for dll in ("cublas64_13.dll", "cublasLt64_13.dll"):
        candidates = [cuda / "bin/x64" / dll, cuda / "bin" / dll]
        entries[native.parent / dll] = next(candidate for candidate in candidates if candidate.is_file())
    src = RUNTIME / "audio.cpp"
    entries[Path("licenses/cuda-13-LICENSE")] = cuda / "LICENSE"
    entries[Path("licenses/audio.cpp-LICENSE")] = src / "LICENSE"
    entries[Path("licenses/five-language-NOTICE")] = RUNTIME / "five-lang-source/NOTICE"
    for directory in (src / "external").iterdir():
        if directory.is_dir():
            for file in directory.iterdir():
                if file.is_file() and file.name.lower().startswith(("license", "copying", "notice")):
                    entries[Path("licenses/audio.cpp") / directory.name / file.name] = file
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        tmp = Path(folder) / path.name
        with ZipFile(tmp, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for rel, file in sorted(entries.items()):
                archive.write(file, (Path("JR-ComfyUI-Drama3") / rel).as_posix())
        with ZipFile(tmp) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Failed ZIP CRC for {bad}")
        tmp.replace(path)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNTIME / "dist/JR-ComfyUI-Drama3-5Lang-Windows.zip")
    package(parser.parse_args().output)
