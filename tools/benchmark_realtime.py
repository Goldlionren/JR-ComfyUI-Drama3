"""Time client-visible streaming audio, then independently transcribe the saved takes."""
# ruff: noqa: E402 -- standalone scripts import the plugin modules from the repository root.

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jr_benchmark_cases import CASES, FIXTURES, REFERENCES
from jr_realtime_client import PORT, RATE, stream_speech

LONG = (
    {
        "id": "zh-long",
        "reference": "zh",
        "seed": 37004,
        "target": "今天早上，我沿着河边的小路走到车站。雨已经停了，地上还有一些积水。我买了一张车票，然后给家里打了一个电话。",
    },
    {
        "id": "en-long",
        "reference": "en",
        "seed": 37005,
        "target": "Before we leave the studio, please close the window and put the microphone back in its case. "
        "I will take the train to the city and call you when I arrive.",
    },
    {
        "id": "mixed-long",
        "reference": "zh",
        "seed": 37006,
        "target": "会议结束以后，请把笔记放到桌上。Please send me the final report. 我收到以后会给你回电话。",
    },
)
SUSTAINED = {
    "id": "mixed-sustained",
    "reference": "zh",
    "seed": 37184,
    "target": "录音开始以前，请先关好窗户，把桌上的水杯放远一点。"
    "Before we begin, take a breath and check that the microphone is ready. "
    "今天我们要从河边出发，穿过公园，然后在车站门口见面。"
    "The first train leaves in the morning, and the last one comes back after dark. "
    "如果下雨，就把蓝色的雨伞带上，把地图放进背包。"
    "When you reach the bridge, look to your left and follow the path beside the water. "
    "路上有一家小书店，橱窗里放着一本关于城市历史的书。"
    "You can stop there for a few minutes and write a note before you continue. "
    "到了车站以后，请把新的时间告诉大家，我们一起决定在哪里吃饭。"
    "At the end of the day, call me from the station so I know you arrived safely. "
    "这些安排都记好了，今天的录音到这里结束。",
}


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def measure(case, seed, path, port):
    pcm = bytearray()
    arrivals = []
    started = time.perf_counter()
    for chunk in stream_speech(case["target"], case["reference"], seed, port=port):
        at = time.perf_counter() - started
        pcm.extend(chunk)
        arrivals.append((at, len(pcm) / (2 * RATE)))
    wall = time.perf_counter() - started
    audio_s = len(pcm) / (2 * RATE)
    playable = next(t for t, seconds in arrivals if seconds >= 0.02)
    # Start a simulated device once 80 ms of PCM has arrived. Later packets must
    # arrive before the previous samples have played; silence within PCM counts as audio.
    first = next(i for i, (_, seconds) in enumerate(arrivals) if seconds >= 0.08)
    playback = arrivals[first][0]
    starvation = max(
        [0.0] + [arrivals[i][0] - playback - arrivals[i - 1][1] for i in range(first + 1, len(arrivals))]
    )
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, RATE, 0, "NONE", ""))
        writer.writeframes(pcm)
    return {
        **case,
        "seed": seed,
        "file": path.name,
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "chunks": len(arrivals),
        "first_pcm_s": arrivals[0][0],
        "playable_20ms_s": playable,
        "playback_80ms_s": playback,
        "underrun_s": starvation,
        "duration_s": audio_s,
        "wall_s": wall,
        "rtf": wall / audio_s,
    }


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def summarize(runs):
    summary = {}
    for lang in ("zh", "en", "mixed"):
        group = [r for r in runs if r["id"].split("-")[0] == lang]
        if not group:
            continue
        summary[lang] = {
            "n": len(group),
            "first_20ms_p95_s": p95([r["playable_20ms_s"] for r in group]),
            "first_80ms_p95_s": p95([r["playback_80ms_s"] for r in group]),
            "rtf_median": statistics.median(r["rtf"] for r in group),
            "rtf_p95": p95([r["rtf"] for r in group]),
            "max_underrun_s": max(r["underrun_s"] for r in group),
        }
    return summary


def run(args):
    import pynvml

    args.output.mkdir(parents=True, exist_ok=True)
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    data = {
        "rate": RATE,
        "repeats": args.repeats,
        "runs": [],
        "gpu": pynvml.nvmlDeviceGetName(gpu),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "protocol": "One request at a time. GET models + POST to last SSE event, 44.1 kHz mono PCM; "
        "start playout with 80 ms accumulated; per-group nearest-rank P95. "
        "VRAM is whole-device usage sampled immediately after each warm take.",
    }
    # Warm the reference and GPU paths. Keep these timings visible but outside warm statistics.
    data["warmup"] = [measure(c, c["seed"], args.output / f"warmup-{c['id']}.wav", args.port) for c in CASES]
    cases = [SUSTAINED] if args.sustained else (*CASES, *LONG)
    for i in range(args.repeats):
        for case in cases:
            path = args.output / f"{case['id']}-{i + 1}.wav"
            row = measure(case, case["seed"] + i * 100, path, args.port)
            row["device_used_gib_after"] = pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**30
            data["runs"].append(row)
            save(args.output / "timings.json", data)
            print(path.name, f"first {row['playable_20ms_s']:.3f}s, RTF {row['rtf']:.3f}", flush=True)
    data["summary"] = summarize(data["runs"])
    data["timing_pass"] = all(
        s["first_20ms_p95_s"] <= 0.5 and s["rtf_p95"] <= 0.7 and s["max_underrun_s"] <= 0.001
        for s in data["summary"].values()
    )
    save(args.output / "timings.json", data)
    pynvml.nvmlShutdown()
    return data["timing_pass"]


def review(out, speaker_path, by_sentence=False, port=PORT):
    import importlib
    from types import ModuleType
    import numpy as np
    import torch
    from jr_script_core import chunk_script

    package = ModuleType("realtime_review")
    package.__path__ = [str(ROOT)]
    sys.modules[package.__name__] = package
    asr = importlib.import_module("realtime_review.jr_analysis_asr")
    quality = importlib.import_module("realtime_review.jr_analysis_quality")
    torch.set_num_threads(6)
    model = asr.whisper_model("", "cpu")
    speaker = quality.speaker_model(speaker_path)
    references = {}
    for key, (filename, _, _, _) in REFERENCES.items():
        with wave.open(str(FIXTURES / filename), "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768
            reference = {
                "waveform": torch.from_numpy(samples.reshape(-1, wav.getnchannels()).T)[None],
                "sample_rate": wav.getframerate(),
            }
        references[key] = quality.speaker_embedding(speaker, asr.mono16(reference))
    from whisper.normalizers import EnglishTextNormalizer

    normalize_en = EnglishTextNormalizer()
    fold = quality.chinese_folder()[1]
    rows = []
    data = json.loads((out / "timings.json").read_text("utf-8"))
    for row in data["runs"]:
        with wave.open(str(out / row["file"])) as w:
            pcm = w.readframes(w.getnframes())
        if hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
            raise ValueError("Audio hash mismatch: " + row["file"])
        audio = {
            "waveform": torch.from_numpy(np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768)[
                None, None
            ],
            "sample_rate": RATE,
        }
        lang = row["id"].split("-")[0]
        mono = asr.mono16(audio)
        cosine = float(torch.dot(quality.speaker_embedding(speaker, mono), references[row["reference"]]))
        segments = []
        if by_sentence:
            offset = 0
            for i, text in enumerate(chunk_script(row["target"], 300)[0]):
                take = b"".join(stream_speech(text, row["reference"], (row["seed"] + i) % 2**32, port=port))
                if pcm[offset : offset + len(take)] != take:
                    raise ValueError("Cannot verify sentence boundaries against the original PCM.")
                start = round(offset / 2 * 16000 / RATE)
                offset += len(take)
                end = round(offset / 2 * 16000 / RATE)
                language = "zh" if any("\u3400" <= c <= "\u9fff" for c in text) else "en"
                heard = model.transcribe(
                    mono[start:end], language=language, fp16=False, temperature=0, verbose=None
                )["text"].strip()
                segments.append({"reference_text": text, "heard": heard, "end_sample": offset // 2})
            if offset != len(pcm):
                raise ValueError("Combined sentence audio length differs from the recorded stream.")
            transcript = " ".join(s["heard"] for s in segments)
        else:
            transcript = model.transcribe(
                mono, language="en" if lang == "en" else "zh", fp16=False, temperature=0, verbose=None
            )["text"].strip()

        # The corpus contains two Chinese numeric expressions with unambiguous readings.
        def tokens(s):
            if lang == "en":
                return normalize_en(s).split()
            normalized = fold(s).replace("9点", "九点").replace("3点", "三点").replace("2楼", "二楼")
            return quality._tokens(normalized)

        wanted, got = tokens(row["target"]), tokens(transcript)
        edits = quality.distance(wanted, got)
        rows.append(
            {
                "file": row["file"],
                "group": lang,
                "target": row["target"],
                "heard": transcript,
                "edits": edits,
                "units": len(wanted),
                "error": edits / len(wanted),
                "tail_matches": wanted[-2:] == got[-2:],
                "speaker_cosine": cosine,
                **({"verified_sentence_boundaries": segments} if by_sentence else {}),
            }
        )
        print(row["file"], ascii(transcript), flush=True)
    summary = {
        lang: sum(r["edits"] for r in rows if r["group"] == lang)
        / sum(r["units"] for r in rows if r["group"] == lang)
        for lang in sorted({r["group"] for r in rows})
    }
    passed = max(summary.values()) <= 0.10 and all(
        r["error"] <= 0.25 and r["tail_matches"] and r["speaker_cosine"] >= 0.55 for r in rows
    )
    save(
        out / ("asr-by-sentence.json" if by_sentence else "asr.json"),
        {
            "recognizer": "local multilingual Whisper small, CPU greedy; "
            + (
                "language by sentence, boundaries matched byte-for-byte to recorded PCM"
                if by_sentence
                else "zh for mixed"
            ),
            "language_error_rates": summary,
            "quality_pass": passed,
            "runs": rows,
        },
    )
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sustained", action="store_true", help="Exercise a continuous bilingual passage.")
    parser.add_argument("--review", action="store_true")
    parser.add_argument(
        "--review-by-sentence",
        action="store_true",
        help="Reproduce each sentence to locate exact boundaries in the saved audio, then run ASR.",
    )
    parser.add_argument("--speaker-model", default="", help="Local CAMPPlus checkpoint for the audio review.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.review or args.review_by_sentence:
        passed = review(args.output, args.speaker_model, args.review_by_sentence, args.port)
    else:
        passed = run(args)
    sys.exit(0 if passed else 1)
