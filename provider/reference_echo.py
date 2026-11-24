"""Detect sustained acoustic copies of a custom reference without a second model."""

import unicodedata

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import soxr

from .realtime_voices import decode_reference


# 8 kHz speech, a 40 ms analysis frame and one feature vector every 40 ms.
# Compare two seconds of changing spectral shape, not just speaker timbre.
RATE = 8000
WINDOW = 50
HANN = np.hanning(320)
FREQUENCIES = np.fft.rfftfreq(512, 1 / RATE)[:, None]
EDGES = 700 * np.expm1(np.linspace(np.log1p(60 / 700), np.log1p(3800 / 700), 26))
MEL = np.maximum(0, np.minimum((FREQUENCIES - EDGES[:-2]) / np.diff(EDGES)[:-1],
                               (EDGES[2:] - FREQUENCIES) / np.diff(EDGES)[1:]))
ORDERS = np.arange(1, 14)
CEPSTRA = (np.sqrt(2 / 24) * np.cos(np.pi / 24 * (np.arange(24)[:, None] + .5) * ORDERS)
           * ORDERS ** .35)


def _moving_average(x, width):
    pad = width // 2
    padded = np.pad(x, ((pad + 1, pad), (0, 0)), mode="edge")
    sums = np.cumsum(padded, axis=0)
    return (sums[width:] - sums[:-width]) / width


def _windows(samples, rate, stride):
    # Overlap the tiles so that a copy crossing an FFT processing boundary
    # still has a complete comparison window. Bound FFT and matrix workspace.
    for offset in range(0, len(samples), 8 * rate):
        tile = samples[offset:offset + 12 * rate]
        if len(tile) < 2 * rate:
            break
        x = soxr.resample(tile.astype(np.float32) / 32768, rate, RATE)
        frames = sliding_window_view(np.pad(x, (160, 160)), 320)[::160]
        power = np.abs(np.fft.rfft(frames * HANN, 512)) ** 2
        coefficients = np.log(np.maximum(power @ MEL, 1e-8)) @ CEPSTRA
        changes = _moving_average(coefficients - _moving_average(coefficients, 75), 3)[::2]
        changes /= np.maximum(np.linalg.norm(changes, axis=1, keepdims=True), 1)
        changes[np.max(np.abs(frames[::2]), axis=1) < .002] = 0
        patterns = sliding_window_view(changes, WINDOW, axis=0)[::stride]
        flat = np.ascontiguousarray(patterns.reshape(len(patterns), -1), dtype=np.float32)
        # Pauses need not dilute a good match, but a mostly silent window
        # cannot establish a copy by a few coincident frames.
        flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), np.sqrt(WINDOW / 2))
        yield flat


def _words(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


class ReferenceEcho:
    def __init__(self, anchors, transcript):
        self.anchors = anchors
        self.transcript = _words(transcript)

    @classmethod
    def from_voice(cls, voice):
        if "voice_ref" not in voice:
            return None
        raw, channels, rate = decode_reference(voice["voice_ref"])
        if rate < 4000 or len(raw) < 2 * rate * channels * 2:
            return None
        mono = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).mean(axis=1, dtype=np.float32)
        anchors = np.concatenate(list(_windows(mono, rate, 10)))
        return cls(anchors, voice["reference_text"])

    def applies(self, dialogue):
        target = _words(dialogue)
        if len(self.transcript) < 4:
            return not self.transcript or self.transcript != target
        return self.transcript not in target

    def matches(self, pcm, rate):
        samples = np.frombuffer(pcm, dtype="<i2")
        return any(np.max(self.anchors @ block.T) >= .86 for block in _windows(samples, rate, 1))
