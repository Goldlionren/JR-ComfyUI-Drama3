"""Isolated GPU A/B test of the official Gated DeltaNet rules versus fla-core."""

# ruff: noqa: E402 -- select the workspace packages and Triton cache before importing torch.
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".runtime/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".runtime/inductor"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import soundfile as sf
import jr_runtime
from jr_runtime.attention import fla_rules
from jr_runtime.model import activate_runtime
from jr_runtime.setup_runtime import FLA_PACKAGE
from jr_teacher import REVISION, validate_teacher


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def test_rules(torch_rules, fast_rules):
    torch.manual_seed(117)
    results = []
    for length in (1, 32, 64):
        op = 1 if length == 1 else 0
        shape = (1, length, 32, 128)
        q, k, v = (
            torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3)
        )
        decay = -torch.rand(shape[:-1], device="cuda")
        beta = torch.rand_like(decay)
        state = torch.randn(1, 32, 128, 128, device="cuda") * 0.02
        args = q, k, v, decay, beta, state
        start = sync_time()
        ref = torch_rules[op](*args)
        sync_time()
        first = sync_time()
        fast = fast_rules[op](*args)
        ready = sync_time()

        def rel(r, t):
            return float(
                (r.float() - t.float()).norm() / r.float().norm().clamp(min=1e-9)
            )

        diffs = [rel(r, t) for r, t in zip(ref, fast, strict=True)]
        if not all(x < 0.03 for x in diffs):
            raise RuntimeError(
                f"FLA shape {length} output/state relative L2 errors: {diffs}"
            )
        timings = {}
        for label, func in (("torch", torch_rules[op]), ("fla", fast_rules[op])):
            values = []
            for _ in range(5):
                tic = sync_time()
                for _ in range(30):
                    func(*args)
                values.append((sync_time() - tic) * 1000 / 30)
            timings[label] = statistics.median(values)
        results.append(
            dict(
                length=length,
                relative_l2_output_state=diffs,
                wall_ms=timings,
                first_fla_seconds=ready - first,
                first_torch_seconds=first - start,
            )
        )
        print("kernel", results[-1], flush=True)
    return results


@contextmanager
def use_rules(layers, funcs):
    counts = Counter()

    def counted(name, op):
        def call(*args, **kw):
            counts[name] += 1
            return op(*args, **kw)

        return call

    with (
        patch.object(layers, "chunk_gated_delta_rule", counted("chunk", funcs[0])),
        patch.object(
            layers, "recurrent_gated_delta_rule", counted("recurrent", funcs[1])
        ),
    ):
        yield counts


@torch.inference_mode()
def benchmark(out, micro_only):
    activate_runtime()
    fla = import_module(FLA_PACKAGE)
    from fish_speech.models.text2semantic.fish_omni import layers

    ops = {
        "torch": (layers.chunk_gated_delta_rule, layers.recurrent_gated_delta_rule),
        "fla": fla_rules(),
    }
    out.mkdir(parents=True, exist_ok=True)
    result = dict(
        revision=REVISION,
        torch=torch.__version__,
        fla=fla.__version__,
        python=platform.python_version(),
        cuda=torch.version.cuda,
        triton=version("triton-windows"),
        gpu=torch.cuda.get_device_name(),
        params=dict(
            mode="line", seed=184, temperature=0.7, top_p=0.9, top_k=0, max_frames=240
        ),
        runs=[],
    )

    def write():
        path = out / ("kernels.json" if micro_only else "results.json")
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(path)

    result["kernels"] = test_rules(ops["torch"], ops["fla"])
    write()
    if micro_only:
        return
    model_dir = validate_teacher(ROOT / ".runtime/teacher")
    start = sync_time()
    _, _, offload = jr_runtime._get_engine(model_dir, "cuda:0", None)
    result["load_seconds_excluding_file_verification"] = sync_time() - start
    result["host_expert_layers"] = offload
    print("teacher ready", offload, flush=True)
    texts = {
        "warmup": "Hi.",
        "zh": "重庆的银行今天九点开门，请到二楼找林老师。",
        "en": "Welcome home.",
    }
    try:
        for case, order in (
            ("warmup", ["torch", "fla"]),
            ("zh", ["torch", "fla", "fla", "torch"]),
            ("en", ["fla", "torch", "torch", "fla"]),
        ):
            for i, backend in enumerate(order, 1):
                notifications, chunks = [], []

                def progress(frame, total):
                    notifications.append(time.perf_counter())

                def stream(audio):
                    chunks.append(time.perf_counter())

                torch.cuda.reset_peak_memory_stats()
                with use_rules(layers, ops[backend]) as calls:
                    tic = sync_time()
                    take = jr_runtime.synthesize(
                        model_dir,
                        texts[case],
                        stream=stream,
                        progress=progress,
                        **result["params"],
                    )
                    elapsed = sync_time() - tic
                n = take["frames"]
                audio = take["audio"]["waveform"].ravel().numpy()
                rate = take["audio"]["sample_rate"]
                if (
                    not calls["chunk"]
                    or not calls["recurrent"]
                    or len(notifications) != n
                    or not chunks
                    or take["finish_reason"] != "eos"
                    or take["host_expert_layers"] != offload
                ):
                    raise ValueError("Incomplete A/B sample")
                filename = f"{case}-{i}-{backend}.wav"
                sf.write(out / filename, audio, rate, subtype="PCM_16")
                row = dict(
                    case=case,
                    backend=backend,
                    text=texts[case],
                    frames=n,
                    audio_seconds=len(audio) / rate,
                    wall_seconds=elapsed,
                    rtf=elapsed / (len(audio) / rate),
                    first_pcm_seconds=chunks[0] - tic,
                    first_frame_seconds=notifications[0] - tic,
                    between_frames_ms=1000
                    * (notifications[-1] - notifications[0])
                    / max(1, n - 1),
                    peak_torch_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    calls=dict(calls),
                    file=filename,
                    sha256=hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                )
                result["runs"].append(row)
                write()
                print(
                    case,
                    i,
                    backend,
                    n,
                    round(elapsed, 2),
                    round(row["rtf"], 3),
                    flush=True,
                )
    finally:
        jr_runtime.unload()


def review_samples(out):
    """Offline Whisper small reads finished takes; speed is summarized per backend and text."""
    from statistics import median

    pkg = ModuleType("jr_fla_validation")
    pkg.__path__ = [str(ROOT)]
    sys.modules[pkg.__name__] = pkg
    asr = import_module(pkg.__name__ + ".jr_analysis_asr")
    quality = import_module(pkg.__name__ + ".jr_analysis_quality")
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    cases = {}
    heard_cache = {}
    for r in results["runs"]:
        if r["case"] == "warmup":
            continue
        record = cases.setdefault(r["case"], {})
        path = out / r["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"Audio checksum mismatch for {path}")
        wave, rate = sf.read(path, dtype="float32")
        audio = {"waveform": torch.from_numpy(wave)[None, None], "sample_rate": rate}
        key = (r["case"], r["sha256"])
        if key not in heard_cache:
            heard_cache[key] = asr.transcribe(
                audio, "Whisper small", device="cpu", language=r["case"]
            )["text"]
        heard = heard_cache[key]
        raw = quality.review(audio, r["text"], heard)
        # The fixed Chinese line contains one numeric spelling variation.
        folded = quality.review(audio, r["text"], heard.replace("9", "九"))
        row = {
            **r,
            "heard": heard,
            "cer": raw["cer"],
            "cer_equating_nine": folded["cer"],
            "wer": raw.get("wer"),
        }
        record.setdefault(r["backend"], []).append(row)
    metrics = (
        "rtf",
        "wall_seconds",
        "first_frame_seconds",
        "first_pcm_seconds",
        "between_frames_ms",
        "audio_seconds",
        "peak_torch_allocated_gib",
    )
    summary = {}
    for case, record in cases.items():
        if any(len(record[b]) != 2 for b in ("torch", "fla")):
            raise ValueError(f"Incomplete two-pair A/B for {case}")
        stats = {
            b: {k: median(r[k] for r in runs) for k in metrics}
            for b, runs in record.items()
        }
        speedup = stats["torch"]["rtf"] / stats["fla"]["rtf"]
        summary[case] = {"median": stats, "rtf_speedup": speedup, "runs": record}
    if set(cases) != {"zh", "en"}:
        raise ValueError("Both cases are required")
    report = dict(
        protocol="Same resident model, shared placement, fixed seed 184, no voice reference. "
        "Both backends warmed up. Chinese ABBA, English BAAB. First PCM uses the node's "
        "48-frame preview (flushed at EOS). Torch allocated memory covers this process only.",
        revision=results["revision"],
        environment={
            "gpu": results["gpu"],
            "torch": results["torch"],
            "fla": results["fla"],
            "triton": version("triton-windows"),
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "host_expert_layers": results["host_expert_layers"],
        },
        summary=summary,
    )
    (out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for case, data in summary.items():
        print(case, data["median"], "RTF speedup", data["rtf_speedup"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fla-benchmark")
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument("--review", action="store_true")
    options = parser.parse_args()
    if options.review:
        review_samples(options.output)
    else:
        benchmark(options.output, options.micro_only)
