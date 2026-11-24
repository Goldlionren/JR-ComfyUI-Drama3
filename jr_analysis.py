"""ComfyUI nodes for offline transcript preparation, review and video dubbing."""

from __future__ import annotations

import json

from comfy_api.latest import io, ui, InputImpl

from . import jr_analysis_asr as asr
from . import jr_analysis_quality as qa
from . import jr_analysis_video as video

CATEGORY = "audio/JR Drama 3/Analysis"
CAST = io.Custom("DRAMA3_CAST")
CUES = io.Custom("JR_DRAMA3_CUES")


def report(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)


def text_input(name, *, optional=False):
    return io.String.Input(name, default="", multiline=True, optional=optional)


def options(name, values):
    return io.Combo.Input(name, options=values)


def string_port(name):
    return io.String.Output(display_name=name)


def schema(node_id, name, description, inputs, outputs):
    return io.Schema(
        node_id=node_id,
        display_name="JR Drama3 · " + name,
        description=description,
        category=CATEGORY,
        inputs=inputs,
        outputs=outputs,
    )


class LocalASR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3LocalASR",
            "离线语音转写",
            "把参考录音转成逐字稿。选择转写模型后会自动查找本地权重。五语合成模型在「Drama 3 · 实时语音」节点选择。",
            [
                io.Audio.Input("audio"),
                io.Combo.Input(
                    "backend", display_name="转写模型", options=["Whisper small", "Paraformer zh"],
                    tooltip="Whisper small 可转写多种语言，Paraformer zh 用于普通话。",
                ),
                options("language", ["auto", "zh", "en"]),
                options("device", ["cpu", "cuda"]),
                io.String.Input(
                    "model_path", display_name="转写模型路径", default="",
                    placeholder="留空自动查找",
                    tooltip="Whisper small 填官方多语种 small.pt 文件的完整路径，默认使用插件目录下的 .runtime/aux/whisper-small.pt。Paraformer zh 填含 model.pt 和 config.yaml 的模型目录。",
                ),
            ],
            [string_port("transcript"), string_port("timed_segments_json")],
        )

    @classmethod
    def execute(cls, audio, backend, language, device, model_path=""):
        data = asr.transcribe(audio, backend, model_path, device, language)
        return io.NodeOutput(data["text"], report(data))


class TranscriptEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3TranscriptEdit",
            "参考逐字稿",
            "可直接填写逐字稿，或连接 ASR 后覆写校订稿。",
            [text_input("edited_text"), io.String.Input("asr_text", force_input=True, optional=True)],
            [string_port("transcript")],
        )

    @classmethod
    def execute(cls, edited_text="", asr_text=""):
        out = edited_text.strip() or (asr_text or "").strip()
        if not out:
            raise ValueError("Provide a corrected transcript or connect ASR text.")
        return io.NodeOutput(out)


class ForcedAlignment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3ForcedAlign",
            "按稿强制对齐",
            "Whisper small 交叉注意力 DTW。短段传音频与稿件，超过 30 秒传附起止时间的稿件段。",
            [
                io.Audio.Input("audio"),
                text_input("transcript"),
                options("language", ["auto", "zh", "en"]),
                options("device", ["cpu", "cuda"]),
                io.String.Input("model_path", default=""),
                text_input("timed_segments_json", optional=True),
            ],
            [string_port("alignment_json"), string_port("subtitle_srt")],
        )

    @classmethod
    def execute(cls, audio, transcript, language, device, model_path="", timed_segments_json=""):
        r = asr.align(audio, transcript, timed_segments_json, model_path, device, language)
        return io.NodeOutput(report(r), video.subtitle_text(r["segments"]))


class WordRegion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3WordRegion",
            "按词选择修补区域",
            "索引从 0 开始，末项包含在内。可手工覆写秒数。",
            [
                text_input("alignment_json"),
                io.Int.Input("first", default=0, min=0, max=1000000),
                io.Int.Input("last", default=0, min=0, max=1000000),
                io.Float.Input("manual_start", default=-1, min=-1, max=86400, step=0.01),
                io.Float.Input("manual_end", default=-1, min=-1, max=86400, step=0.01),
            ],
            [
                io.Float.Output(display_name="start_seconds"),
                io.Float.Output(display_name="end_seconds"),
                string_port("selected_words"),
            ],
        )

    @classmethod
    def execute(cls, alignment_json, first, last, manual_start=-1.0, manual_end=-1.0):
        return io.NodeOutput(*asr.selected_range(alignment_json, first, last, manual_start, manual_end))


class ReadbackCheck(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3ReadbackCheck",
            "读稿质检",
            "将生成音频回读稿与原稿比对；报告错漏片段、可能截断、静音和限次重试信号。",
            [
                io.Audio.Input("audio"),
                text_input("expected_text"),
                text_input("recognized_text"),
                io.Float.Input("silence_dbfs", default=-46.0, min=-120, max=0),
                io.Float.Input("silence_seconds", default=0.8, min=0.02, max=120.0),
                io.Float.Input("max_error_rate", default=0.2, min=0, max=1, step=0.01),
                io.Int.Input("attempt", default=0, min=0, max=100),
                io.Int.Input("retry_limit", default=1, min=0, max=10),
                text_input("asr_segments_json", optional=True),
            ],
            [string_port("review_json"), io.Boolean.Output(display_name="retry_requested")],
        )

    @classmethod
    def execute(
        cls,
        audio,
        expected_text,
        recognized_text,
        silence_dbfs,
        silence_seconds,
        max_error_rate,
        attempt,
        retry_limit,
        asr_segments_json="",
    ):
        r = qa.review(
            audio,
            expected_text,
            recognized_text,
            silence_dbfs,
            silence_seconds,
            max_error_rate,
            attempt,
            retry_limit,
            asr_segments_json,
        )
        return io.NodeOutput(report(r), r["retry_requested"])


class SpeakerCheck(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3SpeakerCheck",
            "声线相似度复核",
            "使用本地 CAMPPlus 做参考声线余弦比较。阈值提示需要人工试听的片段。",
            [
                io.Audio.Input("reference"),
                io.Audio.Input("candidate"),
                io.String.Input("model_path", default=""),
                io.Float.Input("review_threshold", default=0.55, min=-1.0, max=1.0, step=0.01),
            ],
            [
                io.Float.Output(display_name="cosine"),
                io.Boolean.Output(display_name="review_drift"),
                string_port("details"),
            ],
        )

    @classmethod
    def execute(cls, reference, candidate, model_path, review_threshold):
        r = qa.speaker_compare(reference, candidate, model_path, review_threshold)
        return io.NodeOutput(r["cosine"], r["review_speaker_drift"], report(r))


class SpeakerTurns(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3SpeakerTurns",
            "为字幕段分配说话人",
            "在每段一位说话人的前提下，按本地 CAMPPlus 嵌入聚类。重叠对白可人工拆段校订。",
            [
                io.Audio.Input("audio"),
                text_input("timed_segments_json"),
                io.String.Input("model_path", default=""),
                io.Float.Input("match_threshold", default=0.6, min=-1.0, max=1.0, step=0.01),
            ],
            [string_port("speaker_segments_json"), CAST.Output(display_name="reference_cast")],
        )

    @classmethod
    def execute(cls, audio, timed_segments_json, model_path, match_threshold):
        r = qa.label_speakers(audio, timed_segments_json, model_path, match_threshold)
        return io.NodeOutput(report(r), qa.reference_cast(audio, r["segments"]))


class DubbingScript(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3DubbingScript",
            "原轴译配稿",
            "保留时间与角色，支持人工译稿、离线 OPUS 中英互译，也可填写本地 NLLB 模型目录。",
            [
                text_input("timed_segments_json"),
                text_input("translated_lines"),
                io.String.Input("nllb_model_dir", default=""),
                io.String.Input("source_language", default="eng_Latn"),
                io.String.Input("target_language", default="zho_Hans"),
                options("translation_mode", ["manual", "en → zh", "zh → en"]),
            ],
            [
                string_port("speaker_script"),
                string_port("subtitle_srt"),
                string_port("translated_segments_json"),
                CUES.Output(display_name="timed_cues"),
            ],
        )

    @classmethod
    def execute(
        cls,
        timed_segments_json,
        translated_lines,
        nllb_model_dir,
        source_language,
        target_language,
        translation_mode="manual",
    ):
        script, srt, segs = video.dubbing_script(
            timed_segments_json,
            translated_lines,
            nllb_model_dir,
            source_language,
            target_language,
            translation_mode,
        )
        from .jr_delivery_script import cue, validate

        cues = validate(
            [
                cue(
                    i,
                    s["text"],
                    s.get("speaker", ""),
                    start_ms=round(s["start"] * 1000),
                    end_ms=round(s["end"] * 1000),
                )
                for i, s in enumerate(segs, 1)
            ]
        )
        return io.NodeOutput(script, srt, report({"segments": segs}), cues)


class ExtractAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3VideoAudio",
            "视频抽取原声",
            "使用本地 FFmpeg 提取第一条音轨，保留原采样率和声道。",
            [io.Video.Input("video")],
            [io.Audio.Output(display_name="original_audio")],
        )

    @classmethod
    def execute(cls, video):
        from .jr_analysis_video import extract_track

        return io.NodeOutput(extract_track(video))


class Separate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3SeparateStems",
            "分离对白与背景",
            "使用已安装的 GPT-SoVITS BS-RoFormer 本地权重，输出 44.1 kHz 人声与背景。",
            [io.Audio.Input("audio"), options("device", ["cpu", "cuda"])],
            [io.Audio.Output(display_name="vocal"), io.Audio.Output(display_name="background")],
        )

    @classmethod
    def execute(cls, audio, device):
        return io.NodeOutput(*video.separate_stems(audio, device))


class DubMix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3DubMix",
            "对白与背景混音",
            "将对白放到原时间位置，调整背景与对白增益并避免削波。",
            [
                io.Audio.Input("dub"),
                io.Float.Input("voice_db", default=0, min=-80, max=12),
                io.Float.Input("background_db", default=0, min=-80, max=12),
                io.Float.Input("offset_seconds", default=0, min=0, max=86400),
                io.Audio.Input("background", optional=True),
            ],
            [io.Audio.Output(display_name="mixed_audio"), string_port("mix_info")],
        )

    @classmethod
    def execute(cls, dub, voice_db, background_db, offset_seconds, background=None):
        out, r = video.premix(dub, background, voice_db, background_db, offset_seconds)
        return io.NodeOutput(out, report(r))


class ReplaceTrack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JRDrama3ReplaceTrack",
            display_name="JR Drama3 · 按原画面合轨",
            category=CATEGORY,
            description="复制有效视频画面压缩流，用混音替换声音，按画面时长补静音或截尾，输出无损音轨 MKV。",
            inputs=[
                io.Video.Input("original_video"),
                io.Audio.Input("mixed_audio"),
                io.Boolean.Input("save_output", default=True),
                io.String.Input("filename_prefix", default="video/JR-Drama3/dub"),
            ],
            outputs=[io.Video.Output(display_name="dubbed_video")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, original_video, mixed_audio, save_output=True, filename_prefix="video/JR-Drama3/dub"):
        from folder_paths import get_temp_directory, get_output_directory, get_save_image_path

        base = get_output_directory() if save_output else get_temp_directory()
        folder, filename, _, subfolder, _ = get_save_image_path(filename_prefix, base)
        out = video.remux(original_video, mixed_audio, folder, filename)
        location = io.FolderType.output if save_output else io.FolderType.temp
        preview = ui.PreviewVideo([ui.SavedResult(out.name, subfolder, location)])
        return io.NodeOutput(InputImpl.VideoFromFile(str(out)), ui=preview)


class H3Timing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            "JRDrama3H3Timing",
            "将最终台词接入口型工作流",
            "保留台词样本，尾部补静音至 MiniMax H3 的 24 fps、17k+5 帧网格。",
            [io.Audio.Input("audio")],
            [io.Audio.Output(display_name="drive_audio"), io.Int.Output(display_name="h3_frames")],
        )

    @classmethod
    def execute(cls, audio):
        return io.NodeOutput(*video.h3_timing(audio))


NODES = [
    LocalASR,
    TranscriptEdit,
    ForcedAlignment,
    WordRegion,
    ReadbackCheck,
    SpeakerCheck,
    SpeakerTurns,
    DubbingScript,
    ExtractAudio,
    Separate,
    DubMix,
    ReplaceTrack,
    H3Timing,
]
