"""Start or stop the resident Drama 3 streaming engine on port 8195."""
# ruff: noqa: E402 -- also support starting directly from the tools directory.

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .local_service import STATE, local_port, read_record, service_lock, start, stop
else:
    from local_service import STATE, local_port, read_record, service_lock, start, stop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jr_benchmark_cases import FIVE_LANG_FIXTURES, FIVE_LANG_REFERENCES, FIXTURES, REFERENCES
from jr_realtime_client import stream_speech
from jr_realtime_paths import MODELS, engine_binary, model_file
from jr_realtime_paths import RUNTIME as RUNTIME

PORT = 8195
ENGINE_PORT = 8196


@lru_cache(maxsize=32)
def _verify(path, expected, size, mtime):
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")


def verify(path, expected):
    stat = path.stat()
    _verify(path, expected, stat.st_size, stat.st_mtime_ns)


def five_lang_options():
    # Bound the total even on larger development cards. Account for other desktop workloads.
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    free_mib = min(int(s.strip()) for s in result.splitlines() if s.strip())
    budget = int(os.environ.get("JR_DRAMA3_VRAM_GIB", "12"))
    if budget not in (12, 16):
        raise ValueError("JR_DRAMA3_VRAM_GIB must be 12 or 16")
    available = min(free_mib, budget * 1024)
    options = {"fish_drama3.tower_prefill_cache_mb": 64, "fish_drama3.backbone_prefill_cache_mb": 64,
               "fish_drama3.codec_encoder_chunk_samples": 65536}
    if budget == 16 and available >= 15400:
        return {**options, "fish_drama3.disk_experts": False}
    # 40 layers * three 2048 x 512 projections * 110 bytes per 256 weights.
    slots = min(100 if budget == 12 else 160, (available - 6500) // 52)
    if slots < 8:
        raise ValueError("Too little free NVIDIA memory for the five-language model.")
    return {**options, "fish_drama3.disk_experts": True, "fish_drama3.expert_cache_size": slots,
            "fish_drama3.pin_experts": True}


def verified_model(backend):
    model = model_file(backend)
    digest = MODELS[backend][1]
    if digest is None:
        digest = model.with_suffix(".gguf.sha256").read_text("ascii").strip()
    verify(model, digest)
    return model, digest


def command(backend):
    model, digest = verified_model(backend)
    refs = {}
    presets = REFERENCES | FIVE_LANG_REFERENCES if backend == "5lang" else REFERENCES
    for key, (name, sha, transcript, _) in presets.items():
        wav = (FIVE_LANG_FIXTURES if key in FIVE_LANG_REFERENCES else FIXTURES) / name
        verify(wav, sha)
        refs[key] = {"voice_ref": str(wav), "reference_text": transcript}
    if backend != "q4":
        exe = engine_binary(backend)
        config = STATE / f"realtime-{backend}.json"
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
                                **(five_lang_options() if backend == "5lang" else {}),
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


def engine_backend(proc):
    if proc is None:
        return None
    args = proc.cmdline()
    for backend in ("nvfp4", "5lang"):
        if args[-1] == str(STATE / f"realtime-{backend}.json"):
            return backend
    if "serve" in args and Path(args[0]) == engine_binary("q4"):
        return "q4"
    raise RuntimeError("Unrecognized managed engine command.")


def switch_engine(backend):
    with service_lock("realtime.lock"):
        current = read_record("realtime-engine")
        if engine_backend(current) == backend:
            return
        if not current and local_port(ENGINE_PORT):
            raise RuntimeError("The native engine port is in use by another process.")
        # Prepare and verify the requested package before releasing the current engine.
        verified_model(backend)
        if not engine_binary(backend).is_file():
            raise FileNotFoundError(engine_binary(backend))
        args = None if backend == "5lang" else command(backend)
        previous = current.cmdline() if current else None
        env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        if current:
            stop("realtime-engine")
        try:
            # Read the free-memory budget after unloading the old engine.
            if args is None:
                args = command(backend)
            start("realtime-engine", args, env, ENGINE_PORT, "/v1/models")
        except Exception:
            stop("realtime-engine")
            if previous:
                start("realtime-engine", previous, env, ENGINE_PORT, "/v1/models")
            raise


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
            if current and engine_backend(current) != args.backend:
                raise ValueError("Select the model in the node or POST /v1/models/select to switch.")
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
