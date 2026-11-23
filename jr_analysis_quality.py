"""Text, silence and speaker checks; ASR disagreements are review hints."""

from __future__ import annotations

import difflib
from functools import lru_cache
import json
import math
import os
import re
import unicodedata

import numpy as np
import torch

from .jr_analysis_asr import mono16, model_root, ordered_segments, source_path
from .jr_script_core import MARKS


@lru_cache(maxsize=1)
def chinese_folder():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        map_string = ctypes.WinDLL("kernel32", use_last_error=True).LCMapStringEx
        map_string.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.LPCWSTR,
            ctypes.c_int,
            wintypes.LPWSTR,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ssize_t,
        ]
        map_string.restype = ctypes.c_int

        def fold(text):
            if not text:
                return text
            size = map_string("zh-CN", 0x02000000, text, -1, None, 0, None, None, 0)
            if size == 0:
                raise ctypes.WinError(ctypes.get_last_error())
            result = ctypes.create_unicode_buffer(size)
            if not map_string("zh-CN", 0x02000000, text, -1, result, size, None, None, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            return result.value

        return "Windows LCMapStringEx simplified Chinese", fold
    try:
        from opencc import OpenCC
    except ImportError:
        return "raw characters (OpenCC unavailable)", lambda s: s
    return "OpenCC t2s", OpenCC("t2s").convert


def _tokens(text):
    text = unicodedata.normalize("NFKC", MARKS.sub("", text)).casefold().replace("’", "'")
    # Keep every Han character independent and retain letters within English words.
    return re.findall(
        r"[\u3400-\u9fff\U00020000-\U0002fa1f]|[a-z0-9]+(?:'[a-z0-9]+)*|[^\W\d_a-z\u3400-\u9fff\U00020000-\U0002fa1f]",
        text,
    )


def distance(a, b):
    if a == b:
        return 0
    try:
        import editdistance

        return editdistance.eval(a, b)
    except ImportError:
        if len(a) < len(b):
            a, b = b, a
        row = list(range(len(b) + 1))
        for i, left in enumerate(a, 1):
            current = [i]
            for j, right in enumerate(b, 1):
                current.append(min(row[j] + 1, current[-1] + 1, row[j - 1] + (left != right)))
            row = current
        return row[-1]


def acoustic_stats(waveform, threshold_db, pause_s):
    if not -120 <= threshold_db <= 0 or not np.isfinite(pause_s) or pause_s <= 0:
        raise ValueError("Invalid silence settings.")
    peak = np.max(np.abs(waveform))
    wins = 320
    boundaries = np.arange(0, len(waveform), wins)
    counts = np.diff(np.append(boundaries, len(waveform)))
    rms = np.sqrt(np.add.reduceat(waveform.astype(np.float64) ** 2, boundaries) / counts)
    quiet = rms <= 10 ** (threshold_db / 20)
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    silences = [
        {"start": a * wins / 16000, "end": min(b * wins, len(waveform)) / 16000}
        for a, b in edges.reshape(-1, 2)
    ]
    return {
        "duration": len(waveform) / 16000,
        "peak_dbfs": float(20 * np.log10(peak)) if peak > 0 else None,
        "silent": bool(quiet.all()),
        "silence_fraction": float(counts[quiet].sum() / len(waveform)),
        "silences": [s for s in silences if s["end"] - s["start"] >= pause_s],
    }


def review(
    audio,
    expected,
    recognized,
    threshold_db=-46.0,
    pause_s=0.8,
    max_error=0.2,
    attempt=0,
    retry_limit=1,
    alignment=None,
):
    raw_want, raw_got = _tokens(expected), _tokens(recognized)
    norm, fold = chinese_folder()
    want, got = _tokens(fold(expected)), _tokens(fold(recognized))
    if not want:
        raise ValueError("Enter a spoken reference transcript for comparison.")
    if not 0 <= max_error <= 1 or not 0 <= attempt or not 0 <= retry_limit <= 10:
        raise ValueError("Invalid retry or error-rate setting.")
    want_chars = "".join(want).replace("'", "")
    got_chars = "".join(got).replace("'", "")
    raw_ref = "".join(raw_want).replace("'", "")
    raw_hyp = "".join(raw_got).replace("'", "")
    raw_cer = distance(raw_ref, raw_hyp) / len(raw_ref)
    stats = acoustic_stats(mono16(audio), threshold_db, pause_s)
    char_error = (
        raw_cer
        if (want_chars, got_chars) == (raw_ref, raw_hyp)
        else distance(want_chars, got_chars) / len(want_chars)
    )
    word_error = distance(want, got) / len(want)
    has_han = bool(re.search(r"[\u3400-\u9fff\U00020000-\U0002fa1f]", want_chars))
    segments = json.loads(alignment) if isinstance(alignment, str) and alignment.strip() else alignment or []
    if isinstance(segments, dict):
        segments = segments["segments"]
    if not isinstance(segments, list):
        raise ValueError("The ASR report must contain a segment list.")
    word_times = []
    for s in segments:
        word_times.extend([(s["start"], s["end"])] * len(_tokens(fold(s["text"]))))
    # Attach time only when the supplied ASR segments describe the actual recognized text.
    if [t for s in segments for t in _tokens(fold(s["text"]))] != got:
        word_times = []
    diffs = []
    changes = (
        ()
        if want == got
        else difflib.SequenceMatcher(None, want, got, autojunk=len(want) > 1500).get_opcodes()
    )
    for op, ref_start, ref_end, heard_start, heard_end in changes:
        if op == "equal":
            continue
        row = {
            "type": op,
            "expected_indices": [ref_start, ref_end],
            "heard_indices": [heard_start, heard_end],
            "expected": " ".join(want[ref_start:ref_end]),
            "heard": " ".join(got[heard_start:heard_end]),
        }
        if word_times:
            first = max(0, heard_start - 1) if op == "delete" else heard_start
            row["listen_start"] = word_times[min(first, len(word_times) - 1)][0]
            row["listen_end"] = word_times[min(len(word_times) - 1, max(heard_start, heard_end - 1))][1]
        diffs.append(row)
    flags = []
    if stats["silent"]:
        flags.append("silent_audio")
    error_rate = char_error if has_han else word_error
    if error_rate > max_error:
        flags.append("transcript_mismatch")
    if any(d["type"] == "delete" for d in diffs):
        flags.append("possible_omission")
    if diffs and diffs[-1]["type"] == "delete" and diffs[-1]["expected_indices"][1] == len(want):
        flags.append("possible_truncated_tail")
    if stats["silences"]:
        flags.append("review_long_silence")
    # Intentional pauses alone should never trigger a costly re-generation.
    retry = attempt < retry_limit and any(f != "review_long_silence" for f in flags)
    return {
        **stats,
        "expected_transcript": expected,
        "recognized_transcript": recognized,
        "chinese_normalization": norm,
        "raw_cer": raw_cer,
        "cer": char_error,
        "wer" if not has_han else "mixed_token_error_rate": word_error,
        "differences": diffs,
        "flags": flags,
        "retry_requested": retry,
        "attempt": attempt,
        "retry_limit": retry_limit,
    }


def speaker_model(model_path):
    from funasr.models.campplus.model import CAMPPlus

    base = model_root()
    ck = source_path(
        model_path,
        "JR_DRAMA3_CAMPPLUS_PATH",
        [
            base / "asr/campplus_cn_common.bin",
            base / "IndexTTS-2.5/hf_cache/campplus_cn_common.bin",
            base / "IndexTTS-2/campplus_cn_common.bin",
        ],
    )
    model = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
    return model


def speaker_embedding(model, audio16):
    from torchaudio.compliance.kaldi import fbank

    if len(audio16) < 8000 or np.sqrt(np.mean(audio16.astype(np.float64) ** 2)) < 1e-5:
        raise ValueError("Speaker comparison requires at least 0.5 seconds of audible speech.")
    # Sample up to 3 eight-second excerpts across long recordings; avoid averaging silent chunks.
    window = min(8 * 16000, len(audio16))
    starts = np.unique(
        np.linspace(0, len(audio16) - window, min(3, math.ceil(len(audio16) / window)), dtype=int)
    )
    embeddings = []
    with torch.inference_mode():
        for start in starts:
            part = torch.from_numpy(audio16[start : start + window]).unsqueeze(0)
            if part.square().mean().item() < 1e-10:
                continue
            feat = fbank(part, num_mel_bins=80, sample_frequency=16000, dither=0)
            emb = model((feat - feat.mean(0)).unsqueeze(0)).flatten()
            embeddings.append(torch.nn.functional.normalize(emb, dim=0))
    if not embeddings:
        raise ValueError("No usable speech excerpt for comparison.")
    return torch.nn.functional.normalize(torch.stack(embeddings).mean(0), dim=0)


def speaker_compare(reference, candidate, model_path="", threshold=0.55):
    model = speaker_model(model_path)
    a = speaker_embedding(model, mono16(reference))
    b = speaker_embedding(model, mono16(candidate))
    cosine = float(torch.dot(a, b).clamp(-1, 1))
    return {
        "model": "CAMPPlus",
        "cosine": cosine,
        "review_threshold": threshold,
        "review_speaker_drift": cosine < threshold,
        "scope": "speaker similarity for listening review",
    }


def label_speakers(audio, segments, model_path="", threshold=0.60):
    waveform = mono16(audio)
    segs = ordered_segments(segments, len(waveform) / 16000)
    model = speaker_model(model_path)
    centroids = []
    for seg in segs:
        part = waveform[round(seg["start"] * 16000) : round(seg["end"] * 16000)]
        try:
            vector = speaker_embedding(model, part)
        except ValueError:
            seg.update(speaker="unknown", speaker_score=None)
            continue
        scores = [float(torch.dot(vector, torch.nn.functional.normalize(s, dim=0))) for s in centroids]
        j = int(np.argmax(scores)) if scores else None
        if j is None or scores[j] < threshold:
            j = len(centroids)
            centroids.append(vector.clone())
            score = None
        else:
            centroids[j] += vector
            score = scores[j]
        seg.update(speaker=f"S{j + 1}", speaker_score=score)
    return {
        "model": "CAMPPlus",
        "method": "one voice per supplied ASR segment, online cosine clustering",
        "threshold": threshold,
        "segments": segs,
    }


def reference_cast(audio, segments):
    from .jr_script_core import checked_cast

    best = {}
    for s in segments:
        role = s["speaker"]
        if role == "unknown":
            continue
        # For each speaker prefer a complete utterance closest to an 8 s reference;
        # never crop the reference while retaining a mismatched transcript.
        distance_from_target = abs(8 - (s["end"] - s["start"]))
        if role not in best or distance_from_target < best[role][0]:
            best[role] = (distance_from_target, s)
    if len(best) > 16:
        raise ValueError(
            "More than 16 speaker clusters; split the scene or correct the diarization threshold."
        )
    refs = []
    sr = audio["sample_rate"]
    for index, (name, (_, s)) in enumerate(best.items()):
        wave = audio["waveform"][..., round(s["start"] * sr) : round(s["end"] * sr)]
        refs.append(
            dict(
                name=name,
                speaker=index,
                transcript=s["text"],
                language="",
                description="",
                audio={"waveform": wave.clone(), "sample_rate": sr},
            )
        )
    return checked_cast(refs) if refs else []
