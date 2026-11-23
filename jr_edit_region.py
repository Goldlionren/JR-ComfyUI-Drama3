"""Prepare sample-accurate boundaries for the local teacher's audio infill."""
import math

import torch
from comfy_api.latest import io

from .jr_runtime.inference import HOP, RATE, audio_dict, mono_audio


class EditRegion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3EditRegion", display_name="Drama 3 · 波形与补录选区",
            category="audio/JR Drama 3", is_output_node=True,
            description="拖动波形或输入边界，将三路输出接入局部补录。分离的前后音频可直接在接缝插词。",
            inputs=[io.Combo.Input("units", options=["seconds", "samples", "codec_frames", "video_frames", "join_context"]),
                    io.Float.Input("start", default=0, min=0, max=2**40, step=0.001),
                    io.Float.Input("end", default=0, min=0, max=2**40, step=0.001),
                    io.Float.Input("video_fps", default=24, min=1, max=240, step=0.001),
                    io.Audio.Input("source", optional=True),
                    io.Audio.Input("before", optional=True), io.Audio.Input("after", optional=True)],
            outputs=[io.Audio.Output("source_audio"), io.Float.Output("start_seconds"), io.Float.Output("end_seconds")])

    @classmethod
    def execute(cls, units, start, end, video_fps, source=None, before=None, after=None):
        if units == "join_context":
            if before is None or after is None or source is not None:
                raise ValueError("插词接缝模式请分别连接 before 与 after。")
            left, right = mono_audio(before), mono_audio(after)
            wave = torch.cat((left, right))
            first = last = left.numel()
        else:
            if source is None or before is not None or after is not None:
                raise ValueError("按时间或帧选区时请连接 source。")
            wave = mono_audio(source)
            factor = {"seconds": RATE, "samples": 1, "codec_frames": HOP, "video_frames": RATE/video_fps}[units]
            if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start <= end:
                raise ValueError("选区的起止位置无效。")
            if units != "seconds" and any(v != int(v) for v in (start, end)):
                raise ValueError("帧号和采样编号必须为整数。")
            first, last = round(start*factor), round(end*factor)
            if last > wave.numel():
                raise ValueError("补录选区超过音频末尾。")
        # Each displayed column represents the same number of source samples.
        stride = max(1, math.ceil(wave.numel()/600))
        padded = torch.nn.functional.pad(wave, (0, (-wave.numel()) % stride)).view(-1, stride)
        ui = {"region_wave": [{"min": padded.amin(1).tolist(), "max": padded.amax(1).tolist(),
                               "samples": wave.numel(), "stride": stride}]}
        return io.NodeOutput(audio_dict(wave), first/RATE, last/RATE, ui=ui)
