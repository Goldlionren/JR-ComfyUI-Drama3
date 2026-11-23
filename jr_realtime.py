"""ComfyUI node for the resident local quantized Drama 3 service."""

from contextlib import closing
from io import BytesIO
import json
import time
import wave

from comfy import model_management
from comfy_api.latest import io, ui
import numpy as np
import torch

from .jr_delivery_audio import samples
from .jr_generate import CATEGORY, _preview_events, seed_input
from .jr_realtime_client import MAX_REF_BYTES, RATE, stream_speech


class RealtimeSpeech(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JRDrama3Realtime",
            display_name="Drama 3 · 中英实时语音",
            category=CATEGORY,
            is_output_node=True,
            description="本机量化服务逐块生成。voice 选择参考音色，台词可用中文、英文或混读。",
            inputs=[
                io.String.Input("text", multiline=True),
                io.Combo.Input("voice", options=["zh", "en"], tooltip="内置参考音色。连接参考录音后使用录音中的人声。"),
                seed_input(),
                io.Boolean.Input("live", default=True),
                io.String.Input("instructions", display_name="整段语气", default="", optional=True,
                                tooltip="填写简短的表演提示，如 whisper 或 icy calm。留空沿用自然语气。"),
                io.Audio.Input("reference_audio", display_name="参考录音", optional=True,
                               tooltip="自定义音色。连接加载音频或音色库的 AUDIO，并填写匹配的逐字稿。"),
                io.String.Input("reference_text", display_name="参考录音逐字稿", multiline=True, default="",
                                optional=True, tooltip="填写录音中实际说的话。要生成的新台词写在 text。"),
            ],
            outputs=[io.Audio.Output("audio"), io.String.Output("report")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")  # The live listener and resident engine can change between calls.

    @classmethod
    def execute(cls, text, voice, seed, live=True, instructions="", reference_audio=None, reference_text=""):
        reference_text = reference_text.strip()
        if (reference_audio is not None) != bool(reference_text):
            raise ValueError("自定义音色需要同时连接参考录音并填写该录音的逐字稿。")
        preview, finish = _preview_events(live)
        samples = bytearray()
        first_pcm = None
        start = time.perf_counter()
        try:
            wav = _reference_wav(reference_audio) if reference_audio is not None else None
            with closing(stream_speech(text, voice, seed, instructions=instructions,
                                       reference_wav=wav, reference_text=reference_text)) as chunks:
                for pcm in chunks:
                    model_management.throw_exception_if_processing_interrupted()
                    if first_pcm is None:
                        first_pcm = time.perf_counter() - start
                    samples.extend(pcm)
                    if preview:
                        preview({"sample_rate": RATE, "waveform": _wave(bytearray(pcm))})
        finally:
            finish()
        seconds = len(samples) / (2 * RATE)
        report = {
            "first_pcm_seconds": first_pcm,
            "seconds": seconds,
            "rtf": (time.perf_counter() - start) / seconds,
            "voice": "reference_audio" if wav is not None else voice,
            "instructions": instructions,
            "seed": seed,
        }
        audio = {"sample_rate": RATE, "waveform": _wave(samples)}
        return io.NodeOutput(audio, json.dumps(report, ensure_ascii=False), ui=ui.PreviewAudio(audio, cls=cls))


def _wave(pcm):
    return torch.frombuffer(pcm, dtype=torch.int16).reshape(1, 1, -1).float().div_(32768)


def _reference_wav(audio):
    x, rate = samples(audio)
    if len(x) * 2 + 44 > MAX_REF_BYTES:
        raise ValueError("参考 WAV 最大为 5 MiB，请先裁剪参考录音。")
    mono = np.rint(x.mean(axis=1) * 32768).clip(-32768, 32767).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setparams((1, 2, rate, 0, "NONE", ""))
        w.writeframes(mono.tobytes())
    return buffer.getvalue()
