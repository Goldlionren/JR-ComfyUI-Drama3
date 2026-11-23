"""Audio placement, format conversion and streaming mixdown."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import soundfile as sf
import torch


def samples(audio):
    x, rate = audio["waveform"], audio["sample_rate"]
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 3
        or x.shape[0] != 1
        or x.shape[1] not in (1, 2)
        or not x.shape[2]
    ):
        raise ValueError("Each clip must be a mono or stereo AUDIO with one batch item.")
    if type(rate) is not int or not 1000 <= rate <= 384000:
        raise ValueError("Invalid AUDIO sample rate.")
    x = x.detach().to(device="cpu", dtype=torch.float32)[0].T.contiguous().numpy()
    if not np.isfinite(x).all():
        raise ValueError("AUDIO contains non-finite samples.")
    return x, rate


def as_audio(x, rate):
    return {"waveform": torch.from_numpy(np.ascontiguousarray(x.T))[None], "sample_rate": rate}


def ffmpeg():
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise ValueError("Local FFmpeg is required for M4B, format conversion and time stretching.") from exc


def check_cancel():
    from comfy.model_management import throw_exception_if_processing_interrupted

    throw_exception_if_processing_interrupted()


def command(*args):
    return [ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", *map(str, args)]


def convert(args, data=None):
    with subprocess.Popen(
        command(*args),
        stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as proc:
        try:
            while True:
                check_cancel()
                try:
                    output, error = proc.communicate(input=data, timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    data = None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        if proc.returncode:
            raise ValueError("FFmpeg: " + error.decode("utf-8", "replace")[-2000:])
        return output


def audio_filter(x, rate, expression):
    channels = x.shape[1]
    raw = convert(
        [
            "-f",
            "f32le",
            "-ar",
            rate,
            "-ac",
            channels,
            "-i",
            "pipe:0",
            "-af",
            expression,
            "-ar",
            rate,
            "-ac",
            channels,
            "-f",
            "f32le",
            "pipe:1",
        ],
        x.astype("<f4", copy=False).tobytes(),
    )
    return np.frombuffer(raw, dtype="<f4").reshape(-1, channels).copy()


def resample(x, rate, target):
    if target == rate:
        return x
    from scipy.signal import resample_poly

    ratio = Fraction(target, rate)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


@dataclass
class Placement:
    cue: dict
    x: np.ndarray
    start: int
    end: int
    slot_end: int


def arrange(timeline, rate, overflow, max_speed, overlap, gap_ms):
    if not timeline:
        raise ValueError("Timeline is empty.")
    if overflow not in ("error", "extend", "time_stretch") or overlap not in ("mix", "error"):
        raise ValueError("Unknown placement policy.")
    placements = []
    cursor = 0
    for item in timeline:
        check_cancel()
        row = item["cue"]
        x, source_rate = samples(item["audio"])
        edit = item.get("edit", {})
        first = round(edit.get("in_ms", 0) * source_rate / 1000)
        last = round(edit["out_ms"] * source_rate / 1000) if edit.get("out_ms") is not None else len(x)
        if not 0 <= first < last <= len(x):
            raise ValueError(f"Cue {row['id']} has invalid source trim points.")
        x = x[first:last]
        x = resample(x, source_rate, rate)
        start = round(row["start_ms"] * rate / 1000) if row["start_ms"] is not None else cursor
        slot_end = round(row["end_ms"] * rate / 1000) if row["end_ms"] is not None else start + len(x)
        slot = slot_end - start
        if slot <= 0:
            raise ValueError(f"Cue {row['id']} has no duration at the output sample rate.")
        if len(x) > slot:
            if overflow == "error":
                raise ValueError(f"Cue {row['id']} exceeds its slot by {(len(x) - slot) / rate:.3f} s.")
            if overflow == "time_stretch":
                speed = len(x) / slot
                if speed > max_speed:
                    raise ValueError(f"Cue {row['id']} requires {speed:.3f}×; limit is {max_speed:.3f}×.")
                changed = audio_filter(x, rate, f"atempo={speed:.10f}")
                if len(changed) == 0:
                    raise ValueError(f"Cue {row['id']} is too short for pitch-preserving time stretching.")
                x = changed[:slot]
        end = start + len(x)
        placements.append(Placement(row, x, start, end, max(slot_end, end)))
        cursor = max(cursor, max(slot_end, end) + round(gap_ms * rate / 1000))
    if overlap == "error":
        high = 0
        for p in sorted(placements, key=lambda p: p.start):
            if p.start < high:
                raise ValueError(f"Audio overlap at cue {p.cue['id']}.")
            high = p.end
    return placements


def placed_rows(placements, rate):
    return [
        dict(
            p.cue,
            start_ms=round(p.start * 1000 / rate),
            end_ms=max(round(p.start * 1000 / rate) + 1, round(p.slot_end * 1000 / rate)),
        )
        for p in placements
    ]


def blocks(placements, role=None):
    length = max(p.slot_end for p in placements)
    channels = max(p.x.shape[1] for p in placements)
    items = sorted((p for p in placements if role is None or p.cue["role"] == role), key=lambda p: p.start)
    position, active = 0, []
    for start in range(0, length, 65536):
        check_cancel()
        stop = min(start + 65536, length)
        buf = np.zeros((stop - start, channels), dtype=np.float32)
        active = [p for p in active if p.end > start]
        while position < len(items) and items[position].start < stop:
            active.append(items[position])
            position += 1
        for p in active:
            a, b = max(start, p.start), min(stop, p.end)
            if a < b:
                buf[a - start : b - start] += p.x[a - p.start : b - p.start]
        yield buf


def mix_gain(placements):
    peak = max(float(np.max(np.abs(b))) for b in blocks(placements))
    # All stems use the same gain, including stems whose sum cancels.
    # Float stems preserve headroom without clipping individual voices.
    return 1 / max(1.0, peak)


def save_mix(path, placements, rate, gain, role=None):
    length = max(p.slot_end for p in placements)
    channels = max(p.x.shape[1] for p in placements)
    container = "RF64" if length * channels * 4 >= 2**32 - 65536 else "WAV"
    with sf.SoundFile(path, "w", rate, channels, subtype="FLOAT", format=container) as f:
        for b in blocks(placements, role):
            f.write(b * gain)


def postprocess(audio, start_s, end_s, trim_db, fade_in_s, fade_out_s, gain_db, lufs, peak_db, target_rate):
    x, rate = samples(audio)
    start = round(start_s * rate)
    end = round(end_s * rate) if end_s else len(x)
    if start < 0 or end <= start or start >= len(x) or end > len(x):
        raise ValueError("Trim range must be within the input audio; end=0 keeps the tail.")
    x = x[start:end].copy()
    if trim_db > -120:
        audible = np.flatnonzero(np.max(np.abs(x), axis=1) > 10 ** (trim_db / 20))
        if not len(audible):
            raise ValueError("Silence trim would remove the whole clip.")
        x = x[audible[0] : audible[-1] + 1].copy()
    if lufs:
        x = audio_filter(x, rate, f"loudnorm=I={lufs}:TP={peak_db}:LRA=11")
    x *= 10 ** (gain_db / 20)
    for count, reverse in ((round(fade_in_s * rate), False), (round(fade_out_s * rate), True)):
        n = min(len(x), count)
        if n > 0:
            ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
            if reverse:
                x[-n:] *= ramp[::-1]
            else:
                x[:n] *= ramp
    x = resample(x, rate, target_rate or rate)
    peak = float(np.max(np.abs(x)))
    ceiling = math.pow(10, peak_db / 20)
    if peak > ceiling:
        x *= ceiling / peak
    return as_audio(x, target_rate or rate)
