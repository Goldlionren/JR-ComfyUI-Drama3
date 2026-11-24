"""Route ComfyUI production takes to the selected resident GGUF."""
from contextlib import closing
import base64
from dataclasses import dataclass
from io import BytesIO
import time
import wave

import numpy as np
import torch

from .jr_delivery_audio import samples
from .jr_realtime_client import MAX_REF_BYTES, RATE, stream_speech
from .jr_script_core import SPEAKER, checked_cast


@dataclass(frozen=True)
class QuantizedModel:
    id: str
    sha256: str


def pcm_wave(pcm):
    return torch.frombuffer(pcm, dtype=torch.int16).reshape(1, 1, -1).float().div_(32768)


def reference_wav(audio):
    data, rate = samples(audio)
    if len(data) * 2 + 44 > MAX_REF_BYTES:
        raise ValueError("参考 WAV 最大为 5 MiB，请先裁剪参考录音。")
    mono = np.rint(data.mean(axis=1) * 32768).clip(-32768, 32767).astype("<i2")
    buf = BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", ""))
        out.writeframes(mono.tobytes())
    return buf.getvalue()


def render_line(model, text, seed, mode, direction, cast, settings, stream, cancel, previous=None):
    prior = None
    if mode in ("continue", "revise"):
        if model.id == "drama-3-q4":
            raise ValueError("Q4 引擎无法跨请求载入声学历史。请使用 independent、五语、NVFP4 或 FP8。")
        if not previous or previous.get("gguf_sha256") != model.sha256 or previous.get("model") != model.id:
            raise ValueError("请连接同一 GGUF 生成的 history。单次可回传的上一段 WAV 上限为 5 MiB。")
        prior = previous["take"]
    elif mode not in ("line", "design"):
        raise ValueError(f"量化工作流不支持教师的 {mode} 模式。需要原生上下文时请选择 FP8 教师。")
    # The gateway encodes an inline recording as speaker zero. Route each named
    # speaker before sending plain dialogue so another role never gets its voice.
    roles = {}
    for entry in checked_cast(cast):
        speaker = entry["speaker"]
        if speaker in roles and entry["audio"] is not None and roles[speaker]["audio"] is not None:
            raise ValueError(f"GGUF 逐句生成每个角色接收一份参考录音，角色 {speaker} 提供了多份。")
        if speaker not in roles or entry["audio"] is not None:
            roles[speaker] = entry
    parts = SPEAKER.split(text)
    turns = [(0, parts[0]), *zip(map(int, parts[1::2]), parts[2::2])]
    turns = [(speaker, line.strip()) for speaker, line in turns if line.strip()]
    if not turns:
        raise ValueError("台词只有角色标记，请填写要朗读的内容。")
    chunks, segments = [], []
    started = time.monotonic()
    cursor = 0
    for i, (speaker, line) in enumerate(turns):
        cancel()
        role = roles.get(speaker, {})
        audio = role.get("audio")
        ref = reference_wav(audio) if audio is not None else None
        voice = "zh" if ref is not None else "none"
        cue = " ".join(filter(None, [role.get("description", ""), direction])).strip()
        start = cursor
        with closing(stream_speech(line, voice, (seed + i) % 2**32, model=model.id,
                     reference_wav=ref, reference_text=role.get("transcript", "") if ref else "",
                     instructions=cue, sampling=settings, previous_take=prior)) as response:
            for pcm in response:
                cancel()
                clip = pcm_wave(bytearray(pcm))
                chunks.append(clip)
                cursor += clip.shape[-1]
                if stream:
                    stream({"waveform": clip, "sample_rate": RATE})
        segments.append({"speaker": speaker, "text": line, "start_sample": start, "end_sample": cursor})
    if not cursor:
        raise ValueError("量化引擎未返回语音。")
    waveform = torch.cat(chunks, dim=-1)
    if not torch.any(waveform):
        from .jr_jobs import IncompleteTakeError
        raise IncompleteTakeError("量化引擎返回静音，请检查台词、参考音或种子。")
    elapsed = time.monotonic() - started
    output = {"waveform": waveform, "sample_rate": RATE}
    history = None
    if model.id != "drama-3-q4" and cursor * 2 + 44 <= MAX_REF_BYTES and len(text) <= 4096:
        encoded = base64.b64encode(reference_wav(output)).decode("ascii")
        history = {"model": model.id, "gguf_sha256": model.sha256,
                   "take": {"text": text, "wav": {"type": "base64", "data": encoded}}}
    return output, history, {
        "model": model.id, "revision": model.sha256, "mode": mode, "seed": seed,
        "sampling": settings, "finish_reason": "complete", "seconds": cursor / RATE,
        "elapsed_seconds": elapsed, "real_time_factor": elapsed * RATE / cursor, "segments": segments,
    }
