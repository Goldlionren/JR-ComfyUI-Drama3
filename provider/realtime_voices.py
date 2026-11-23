"""Adapt versioned ComfyUI voices and inline WAV references to the quantized engine."""

import base64
import io
import struct
import wave

import numpy as np

from jr_realtime_client import MAX_REF_BYTES, is_saved_voice
from jr_voice_store import load_voice, save_voice


def reference_text(value):
    if not isinstance(value, str) or not 0 < len(value.strip()) <= 4096:
        raise ValueError("Provide the exact reference_text for the recording (1–4096 characters).")
    return value.strip()


def decode_reference(ref):
    if not isinstance(ref, dict) or ref.get("type") != "base64" or not isinstance(ref.get("data"), str):
        raise ValueError("voice_ref must contain type=base64 and PCM WAV data.")
    data = ref["data"]
    if len(data) > 4 * ((MAX_REF_BYTES + 2) // 3):
        raise ValueError("Reference WAV exceeds 5 MiB.")
    raw = base64.b64decode(data, validate=True)
    if len(raw) > MAX_REF_BYTES:
        raise ValueError("Reference WAV exceeds 5 MiB.")
    try:
        with wave.open(io.BytesIO(raw)) as w:
            frames, channels, rate = w.getnframes(), w.getnchannels(), w.getframerate()
            if not frames or channels not in (1, 2) or w.getsampwidth() != 2 or not rate:
                raise ValueError("Expected a nonempty mono or stereo PCM16 WAV.")
            pcm = w.readframes(frames)
            if len(pcm) != frames * channels * 2:
                raise ValueError("Reference WAV is truncated.")
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError("Invalid reference WAV.") from exc
    return pcm, channels, rate


def register_voice(db, body):
    fields = {"name", "language", "description", "tags", "voice_ref", "reference_text"}
    if not isinstance(body, dict) or set(body) - fields:
        raise ValueError("Expected a voice registration with name, voice_ref and reference_text.")
    metadata = {key: body.get(key, "") for key in ("name", "language", "description", "tags")}
    if not all(isinstance(value, str) for value in metadata.values()) or not metadata["name"].strip():
        raise ValueError("Supply a nonempty voice name and text metadata.")
    meta = {key: value.strip() for key, value in metadata.items()}
    transcript = reference_text(body.get("reference_text"))
    pcm, channels, rate = decode_reference(body.get("voice_ref"))
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).T[None].astype("float32") / 32768
    version = save_voice(db, **meta, transcript=transcript, waveform=samples, rate=rate)
    return {"id": version, "object": "voice", "source": "library", **meta}


def saved_options(db, version):
    samples, row = load_voice(db, version)
    transcript = reference_text(row["transcript"])
    if samples.ndim != 3 or samples.shape[0] != 1 or samples.shape[1] not in (1, 2) or not samples.shape[2]:
        raise ValueError("Select a nonempty mono or stereo reference in the voice library.")
    if samples.size * 2 + 44 > MAX_REF_BYTES:
        raise ValueError("Saved reference exceeds the realtime limit of 5 MiB. Save a shorter recording.")
    pcm = np.rint(samples[0].T * 32768).clip(-32768, 32767).astype("<i2")
    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setparams((samples.shape[1], 2, row["sample_rate"], 0, "NONE", ""))
        w.writeframes(pcm.tobytes())
    data = base64.b64encode(wav.getvalue()).decode("ascii")
    return {"voice_ref": {"type": "base64", "data": data}, "reference_text": transcript}


def voice_options(body, backend, db=None):
    voice = body.get("voice", "zh")
    if is_saved_voice(voice):
        if backend != "nvfp4":
            raise ValueError("Select NVFP4 to use a saved voice ID.")
        if "voice_ref" in body or "reference_text" in body:
            raise ValueError("A saved voice ID already supplies its reference recording and transcript.")
        return saved_options(db, voice)
    if voice not in ("zh", "en"):
        raise ValueError("Choose zh, en or an exact saved voice ID from /v1/audio/voices.")
    if "voice_ref" not in body:
        if body.get("reference_text", "") != "":
            raise ValueError("reference_text needs a matching voice_ref recording.")
        return {"voice": voice}
    if backend != "nvfp4":
        raise ValueError("Select the NVFP4 backend to use a custom reference recording.")
    transcript = reference_text(body.get("reference_text"))
    ref = body["voice_ref"]
    decode_reference(ref)
    return {"voice_ref": {"type": "base64", "data": ref["data"]}, "reference_text": transcript}
