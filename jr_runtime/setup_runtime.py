"""Prepare the pinned Fish inference source and DAC wheel below .runtime.

Run with the Python used by ComfyUI. Use --offline when the checkout and
wheel have been copied from another machine. No weights are downloaded.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REVISION = "0f8a12e8a651062fdee9c33a894bdccd3a8513f1"
DAC_WHEEL = "descript_audio_codec-1.0.0-py3-none-any.whl"
DAC_SHA256 = "7ecd1883040613d143494f27f199c11c7053a8a7f9fd2120568c5a98db926e61"
FLA_WHEEL = "fla_core-0.6.0-py3-none-any.whl"
FLA_SHA256 = "b0708b70e0b119cd58a0a5146ea56cb175cef6ed969d8ac60267e3e077ce3ebd"
FLA_PACKAGE = "jr_drama3_fla_060"


def runtime_dir():
    return Path(os.environ.get("JR_DRAMA3_RUNTIME", ROOT / ".runtime")).resolve()


def setup_fla(offline):
    """Prepare the pinned FLA kernels under a private name shared only by JR nodes."""
    base = runtime_dir()
    vendor = base / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    target = vendor / FLA_PACKAGE
    if target.is_dir():
        return target
    wheels = base / "wheels"
    wheel = wheels / FLA_WHEEL
    if not wheel.is_file() and not offline:
        wheels.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps",
                        "--only-binary=:all:", "--dest", str(wheels), "fla-core==0.6.0"], check=True)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != FLA_SHA256:
        raise RuntimeError(f"FLA wheel digest mismatch: {wheel}")
    # Stage alongside the installed package. On Windows a TemporaryDirectory
    # has an owner-only ACL that can survive rename and hide ops from ComfyUI.
    package = vendor / f".{FLA_PACKAGE}-{uuid.uuid4().hex}"
    package.mkdir()
    try:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if not name.startswith("fla/") or name.endswith("/"):
                    continue
                path = package / name.removeprefix("fla/")
                path.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(name)
                # These public re-exports eagerly import training-only Inductor modules.
                # The GDN implementation imports its inference dependencies directly.
                if name in ("fla/modules/__init__.py", "fla/ops/__init__.py"):
                    data = b""
                elif path.suffix == ".py":
                    data = re.sub(rb"\bfla(?=\.|\s+import\b)", FLA_PACKAGE.encode(), data)
                path.write_bytes(data)
            (package / "LICENSE").write_bytes(archive.read("fla_core-0.6.0.dist-info/licenses/LICENSE"))
        try:
            package.rename(target)
        except FileExistsError:
            if not target.is_dir():
                raise
    finally:
        if package.exists() and package.resolve().parent == vendor.resolve():
            shutil.rmtree(package)
    return target


def setup(offline=False):
    base = runtime_dir()
    base.mkdir(parents=True, exist_ok=True)
    src = base / "fish-speech"
    if not src.exists() and not offline:
        with tempfile.TemporaryDirectory(dir=base, prefix="fish-source-") as temporary:
            staging = Path(temporary) / "checkout"
            subprocess.run(["git", "init", str(staging)], check=True)
            subprocess.run(["git", "-C", str(staging), "fetch", "--depth=1",
                            "https://github.com/fishaudio/fish-speech.git", REVISION], check=True)
            subprocess.run(["git", "-C", str(staging), "checkout", "--detach", "FETCH_HEAD"], check=True)
            if not src.exists():
                staging.rename(src)
    git = ["git", "-c", f"safe.directory={src.as_posix()}", "-C", str(src)]
    rev = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise RuntimeError(f"Expected fish-speech {REVISION}, found {rev} in {src}.")
    wheels = base / "wheels"
    wheel = wheels / DAC_WHEEL
    if not wheel.exists() and not offline:
        wheels.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps",
                        "--only-binary=:all:", "--dest", str(wheels),
                        "descript-audio-codec==1.0.0"], check=True)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != DAC_SHA256:
        raise RuntimeError(f"DAC wheel digest mismatch: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(base / "vendor")
    # The Aki runtime already contains the dependencies below. For smaller
    # installations add missing Python packages locally without replacing torch.
    sys.path.insert(0, str(base / "vendor"))
    packages = {"hydra": "hydra-core", "loguru": "loguru", "audiotools": "descript-audiotools",
                "argbind": "argbind", "tokenizers": "tokenizers", "safetensors": "safetensors",
                "einops": "einops", "tqdm": "tqdm", "scipy": "scipy"}
    missing = [package for name, package in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing inference dependencies in {sys.executable}: {', '.join(missing)}. "
                           f"Install their compatible wheels into {base / 'vendor'} or the isolated runtime.")
    print(f"Fish source {rev} and verified DAC wheel ready at {base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fla", action="store_true", help="Prepare only the optional FLA inference kernels.")
    args = parser.parse_args()
    if args.fla:
        print(f"FLA ready at {setup_fla(args.offline)}")
    else:
        setup(args.offline)
