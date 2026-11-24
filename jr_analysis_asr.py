"""Offline reference ASR and forced alignment against the supplied words."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .jr_delivery_audio import resample, samples

ROOT = Path(__file__).resolve().parent
WHISPER_SHA = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
AUX = ROOT / ".runtime" / "aux"
WHISPER_SETUP = "在插件目录运行 python -B jr_analysis_prepare.py whisper，然后清空转写模型路径。"
LOCAL_GPT = Path("F:/GPT-SoVITS-v2-240821")
PARAFORMER = LOCAL_GPT / "tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


def source_path(value, variable, candidates, *, directory=False):
    name = str(value or os.environ.get(variable, "")).strip().strip('"')
    paths = [Path(name).expanduser()] if name else list(candidates)
    for path in paths:
        if str(path).startswith(("\\\\", "//")) or "://" in str(path):
            raise ValueError("Select a local model on disk.")
        if path.is_dir() if directory else path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Local resource missing. Set {variable} or supply its local path. Checked: {paths}"
    )


def model_root():
    try:
        import folder_paths

        return Path(folder_paths.models_dir)
    except ImportError:
        return AUX


def asr_path(value, variable, candidates, *, directory=False):
    name = (str(value or "").strip() or os.environ.get(variable, "")).strip().strip('"')
    if Path(name).suffix.lower() == ".gguf":
        raise ValueError(
            "GGUF 用于语音合成。请在「Drama 3 · 实时语音」的 model 中选择对应模型。"
            "本节点转写参考录音，请清空 model_path 自动查找，或填写 Whisper small 的 .pt 文件"
            "（Paraformer zh 填完整模型目录）。"
        )
    try:
        return source_path(name, variable, candidates, directory=directory)
    except FileNotFoundError:
        checked = name or "、".join(map(str, candidates))
        hint = "请填写完整的 Paraformer zh 模型目录。" if directory else WHISPER_SETUP
        raise FileNotFoundError(f"找不到转写模型，已检查 {checked}。{hint}") from None


def mono16(audio):
    x, rate = samples(audio)
    return np.ascontiguousarray(resample(x.mean(axis=1), rate, 16000), dtype=np.float32)


@lru_cache(maxsize=8)
def _digest(filename, size, mtime):
    with open(filename, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def whisper_model(model_path, device):
    path = asr_path(
        model_path,
        "JR_DRAMA3_WHISPER_PATH",
        [
            AUX / "whisper-small.pt",
            model_root() / "whisper/small.pt",
            Path.home() / ".cache/whisper/small.pt",
        ],
    )
    stat = path.stat()
    checksum = _digest(str(path), stat.st_size, stat.st_mtime_ns)
    if checksum != WHISPER_SHA:
        raise ValueError(
            f"Whisper small 权重校验失败：{path}。实际 SHA256 {checksum}，"
            f"官方多语种 small.pt 应为 {WHISPER_SHA}。{WHISPER_SETUP}"
        )
    import whisper

    # A full existing file path bypasses Whisper's named-model downloader.
    model = whisper.load_model(str(path), device=device).eval()
    model.set_alignment_heads(whisper._ALIGNMENT_HEADS["small"])
    return model


def transcribe(audio, backend, model_path="", device="cpu", language="auto"):
    waveform = mono16(audio)
    if backend == "Whisper small":
        model = whisper_model(model_path, device)
        r = model.transcribe(
            waveform,
            language=None if language == "auto" else language,
            fp16=device.startswith("cuda"),
            temperature=0,
            verbose=None,
            task="transcribe",
        )
        segments = []
        duration = len(waveform) / 16000
        for s in r["segments"]:
            text = s["text"].strip()
            if not text:
                continue
            start = max(float(s["start"]), segments[-1]["end"] if segments else 0)
            end = min(float(s["end"]), duration)
            if start < end:
                segments.append({"start": start, "end": end, "text": text})
            elif segments:
                segments[-1]["text"] += " " + text
        return {
            "model": "OpenAI Whisper small",
            "language": r["language"],
            "text": r["text"].strip(),
            "timing": "Whisper predicted segment timestamps",
            "segments": segments,
        }
    if backend != "Paraformer zh":
        raise ValueError("Unknown local ASR backend")
    path = asr_path(
        model_path,
        "JR_DRAMA3_PARAFORMER_DIR",
        [model_root() / "asr/paraformer-zh", PARAFORMER],
        directory=True,
    )
    for name in ("model.pt", "tokens.json", "config.yaml", "am.mvn", "configuration.json"):
        if not (path / name).is_file():
            raise ValueError(f"Incomplete local Paraformer model: {name}")
    if language == "en":
        raise ValueError("For English use Whisper small.")
    from funasr import AutoModel

    model = AutoModel(
        model=str(path),
        device=device,
        disable_update=True,
        check_latest=False,
        trust_remote_code=False,
        local_files_only=True,
        disable_pbar=True,
    )
    if len(waveform) > 16000 * 60:
        raise ValueError(
            "Split clips longer than 60 seconds for this Paraformer checkpoint, or use Whisper small."
        )
    result = model.generate(input=waveform, fs=16000, disable_pbar=True)[0]["text"].strip()
    return {
        "model": "FunASR Paraformer zh",
        "language": "zh",
        "text": result,
        "timing": "whole input clip; timestamps require the alignment node",
        "segments": [{"start": 0.0, "end": len(waveform) / 16000, "text": result}] if result else [],
    }


def read_segments(value):
    obj = json.loads(value) if isinstance(value, str) else value
    segs = obj.get("segments") if isinstance(obj, dict) else obj
    if not isinstance(segs, list) or not segs:
        raise ValueError("Expected a nonempty list of start/end/text segments.")
    return segs


def ordered_segments(value, duration=None, max_seconds=None):
    segs = read_segments(value)
    previous = 0.0
    out = []
    for seg in segs:
        start, end = float(seg["start"]), float(seg["end"])
        text = seg["text"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or not np.isfinite([start, end]).all()
            or start < previous
            or end <= start
        ):
            raise ValueError("Segments must be ordered, non-overlapping and contain text.")
        if duration is not None and end > duration + 0.035:
            raise ValueError("A time range falls beyond the end of the recording.")
        if max_seconds is not None and end - start > max_seconds + 0.0001:
            raise ValueError(f"Split transcript segments longer than {max_seconds} seconds.")
        out.append(
            {
                **seg,
                "text": text.strip(),
                "start": start,
                "end": min(end, duration) if duration is not None else end,
            }
        )
        previous = end
    return out


def align(audio, text, segments_json="", model_path="", device="cpu", language="auto"):
    import whisper
    from whisper.tokenizer import get_tokenizer
    from whisper.timing import find_alignment, merge_punctuations

    waveform = mono16(audio)
    duration = len(waveform) / 16000
    if segments_json.strip():
        segments = ordered_segments(segments_json, duration, 30.0)

        def strip(s):
            return "".join(s.split()).casefold()

        if text.strip() and strip("".join(s["text"] for s in segments)) != strip(text):
            raise ValueError(
                "The edited transcript and segment texts differ. Correct segment text or leave transcript blank."
            )
    else:
        if duration > 30 or not text.strip():
            raise ValueError(
                "Provide matching start/end/text segments for long recordings, or provide text for a clip up to 30 s."
            )
        segments = [{"start": 0.0, "end": duration, "text": text.strip()}]
    model = whisper_model(model_path, device)
    words = []
    with torch.inference_mode():
        for s in segments:
            chunk = waveform[round(s["start"] * 16000) : round(s["end"] * 16000)]
            if len(chunk) < 1600:
                raise ValueError("Alignment requires at least 0.1 s of audio in each segment.")
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(chunk), n_mels=model.dims.n_mels).to(device)
            lang = language
            if lang == "auto":
                _, probabilities = model.detect_language(mel)
                lang = max(probabilities, key=probabilities.get)
            tokenizer = get_tokenizer(
                True, num_languages=model.num_languages, language=lang, task="transcribe"
            )
            aligned_text = s["text"] if lang in ("zh", "ja", "th", "lo", "my", "yue") else " " + s["text"]
            tokens = tokenizer.encode(aligned_text)
            if len(tokens) + len(tokenizer.sot_sequence) + 2 > model.dims.n_text_ctx:
                raise ValueError(
                    "Transcript segment exceeds Whisper's text context. Divide it at a known boundary."
                )
            timing = find_alignment(model, tokenizer, tokens, mel, max(2, len(chunk) // 160))
            if not timing:
                raise ValueError("The model returned no alignment. Check the transcript and recording.")
            merge_punctuations(timing, "\"'“¿([{-", "\"'.。,，!！?？:：”)]}、")
            offset = len(words)
            for w in timing:
                if not w.word.strip():
                    continue
                words.append(
                    {
                        "index": len(words),
                        "text": w.word,
                        "start": min(s["end"], s["start"] + w.start),
                        "end": min(s["end"], s["start"] + w.end),
                        "confidence": float(w.probability),
                    }
                )
            if offset == len(words):
                raise ValueError("The segment did not contain an alignable word.")
            s["input_start"], s["input_end"] = s["start"], s["end"]
            s["start"], s["end"] = words[offset]["start"], words[-1]["end"]
            s["language"] = lang
    return {
        "model": "OpenAI Whisper small",
        "method": "forced transcript, cross-attention DTW, 20 ms grid",
        "granularity": "English words / Chinese tokenizer units (one or more characters), no phoneme timing",
        "segments": segments,
        "words": words,
    }


def selected_range(alignment, first, last, manual_start=-1, manual_end=-1):
    words = json.loads(alignment)["words"]
    if not 0 <= first <= last < len(words):
        raise ValueError(f"Select inclusive 0-based word indices in 0..{len(words) - 1}.")
    start = float(words[first]["start"] if manual_start < 0 else manual_start)
    end = float(words[last]["end"] if manual_end < 0 else manual_end)
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError("Selected words have no positive duration; adjust the boundaries.")
    return start, end, "".join(w["text"] for w in words[first : last + 1]).strip()
