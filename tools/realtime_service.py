"""Start or stop the resident Drama 3 streaming engine on port 8195."""
# ruff: noqa: E402 -- also support starting directly from the tools directory.

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from local_service import STATE, local_port, read_record, service_lock, start, stop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jr_benchmark_cases import FIXTURES, REFERENCES
from jr_realtime_client import stream_speech

RUNTIME = ROOT / ".runtime"
PORT = 8195
ENGINE_PORT = 8196
MODELS = {
    "q4": ("Drama-35B-A3B-v3-Q4_K.gguf", "2a57b14c9b208a2ff833ab7305b920ca8f154ad807004a52d96a86a90a55523e"),
    "nvfp4": (
        "Drama3-V2-78e5ab7-NVFP4-256E.gguf",
        "5577beb99be7bc0f4c8324daea90e2f3c8765d17842db41016560fb4820312b1",
    ),
}


def verify(path, expected):
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")


def engine_binary(backend):
    if backend == "q4":
        return RUNTIME / "speechcpp" / ("speech.exe" if os.name == "nt" else "speech")
    build = "windows-cuda-release" if os.name == "nt" else "linux-cuda-release"
    binary = "audiocpp_server.exe" if os.name == "nt" else "audiocpp_server"
    return RUNTIME / "audio.cpp" / "build" / build / "bin" / binary


def model_file(backend):
    override = os.environ.get("JR_DRAMA3_NVFP4_MODEL") if backend == "nvfp4" else None
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError("JR_DRAMA3_NVFP4_MODEL must be an absolute path to the GGUF file.")
        return path
    return RUNTIME / MODELS[backend][0]


def command(backend):
    model = model_file(backend)
    digest = MODELS[backend][1]
    verify(model, digest)
    refs = {}
    for key, (wav, sha, transcript, _) in REFERENCES.items():
        verify(FIXTURES / wav, sha)
        refs[key] = {"voice_ref": str(FIXTURES / wav), "reference_text": transcript}
    if backend == "nvfp4":
        exe = engine_binary(backend)
        config = STATE / "realtime-nvfp4.json"
        config.write_text(
            json.dumps(
                {
                    "host": "127.0.0.1",
                    "port": ENGINE_PORT,
                    "backend": "cuda",
                    "threads": 8,
                    "lazy_load": False,
                    "models": [
                        {
                            "id": "drama-3",
                            "family": "fish_drama3",
                            "path": str(model),
                            "task": "tts",
                            "mode": "streaming",
                            "voice_presets": refs,
                            "session_options": {
                                "fish_drama3.cpu_expert_layers": 0,
                                "fish_drama3.disk_experts": False,
                                "fish_drama3.perf_mode": "cuda_perf",
                                "fish_drama3.context_length": 2048,
                            },
                            "default_request_options": {
                                "max_tokens": 920,
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "top_k": 30,
                            },
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return [str(exe), "--config", str(config)]

    exe = engine_binary(backend)
    devices = subprocess.check_output([str(exe), "devices"], text=True)
    device = next((line.split()[0] for line in devices.splitlines() if "NVIDIA" in line), None)
    if not device:
        raise RuntimeError(f"No NVIDIA Vulkan device found:\n{devices}")
    args = [
        str(exe),
        "serve",
        str(model),
        "--device",
        device,
        "--host",
        "127.0.0.1",
        "--port",
        str(ENGINE_PORT),
        "--temperature",
        "0.7",
        "--top-p",
        "0.9",
        "--top-k",
        "30",
    ]
    for key, ref in refs.items():
        sha = REFERENCES[key][1]
        cached = STATE / f"q4-{digest[:12]}-{sha[:12]}.voice.gguf"
        if not cached.exists():
            tmp = cached.with_suffix(".tmp.gguf")
            subprocess.run(
                [
                    str(exe),
                    "voice",
                    str(model),
                    ref["voice_ref"],
                    str(tmp),
                    "--transcript",
                    ref["reference_text"],
                    "--device",
                    "cpu",
                ],
                check=True,
            )
            tmp.replace(cached)
        args += ["--add-voice", f"{key}={cached}"]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("start", "stop", "status"))
    ap.add_argument("--backend", choices=MODELS, default="nvfp4")
    args = ap.parse_args()
    STATE.mkdir(exist_ok=True, parents=True)
    with service_lock("realtime.lock"):
        current = read_record("realtime-engine")
        gateway = read_record("realtime")
        if args.action == "stop":
            stop("realtime")
            stop("realtime-engine")
        elif args.action == "status":
            print(f"{current.name()} PID {current.pid}" if current else "engine stopped")
            print(f"HTTP PID {gateway.pid}, port {PORT}" if gateway else "HTTP stopped")
        else:
            if not gateway and local_port(PORT):
                raise RuntimeError(f"Port {PORT} is already in use.")
            expected = engine_binary(args.backend)
            if (current and Path(current.cmdline()[0]) != expected) or (
                gateway and gateway.cmdline()[-1] != args.backend
            ):
                raise ValueError("A different realtime engine is running. Stop it before switching.")
            if not current and local_port(ENGINE_PORT):
                raise RuntimeError(f"Port {ENGINE_PORT} is already in use.")
            env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
            engine_cmd = current.cmdline() if current else command(args.backend)
            try:
                start("realtime-engine", engine_cmd, env, ENGINE_PORT, "/v1/models")
                if not current:
                    for text, voice in (("你好。", "zh"), ("Hello.", "en")):
                        for _ in stream_speech(text, voice, 184, port=ENGINE_PORT):
                            pass
                start(
                    "realtime",
                    [sys.executable, "-B", "-u", "-m", "provider.realtime", "--backend", args.backend],
                    env,
                    PORT,
                    "/health",
                )
            except Exception:
                if not gateway:
                    stop("realtime")
                if not current:
                    stop("realtime-engine")
                raise


if __name__ == "__main__":
    main()
