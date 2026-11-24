"""ComfyUI boundaries for the pinned local Drama 3 model."""
from __future__ import annotations

import base64
import json
import time
import uuid

import torch
from comfy_api.latest import io
from comfy import model_management
from comfy.utils import ProgressBar
from .jr_teacher import MANIFEST, REVISION, FlaTeacherPath, locate_teacher, validate_teacher
from .jr_jobs import IncompleteTakeError, cast_key, key_for, read_take, require_complete, save_take
from .jr_gguf import QuantizedModel, render_line as render_quantized
from .jr_realtime_client import IncompleteSpeechError, installed_model, unload_model
from .jr_realtime_paths import PUBLIC_MODELS

MODEL = io.Custom("JR_DRAMA3_TEACHER")
HISTORY = io.Custom("JR_DRAMA3_HISTORY")
CAST = io.Custom("DRAMA3_CAST")
SETTINGS = io.Custom("JR_DRAMA3_SAMPLING")
CUES = io.Custom("JR_DRAMA3_CUES")
TAKES = io.Custom("JR_DRAMA3_TAKES")
CATEGORY = "audio/JR Drama 3"
DEFAULTS = {"temperature": 0.7, "top_p": 0.9, "top_k": 0, "max_frames": 1024}
MAX_SEED = 2**32 - 1


class EmptyTakeError(IncompleteTakeError):
    """Sampling completed without usable PCM."""


def seed_input():
    return io.Int.Input("seed", min=0, max=MAX_SEED, default=0, control_after_generate=True)


class Teacher(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3Teacher", display_name="Drama 3 · 语音模型（FP8 / GGUF）",
                         category=CATEGORY, inputs=[io.String.Input("directory", default="",
                         tooltip="空白使用已安装的新版教师。可填写本机模型目录。"),
                         io.Boolean.Input("use_fla", display_name="FLA 加速", default=False, optional=True,
                             label_on="开启", label_off="关闭",
                             tooltip="仅对 FP8 教师生效。"),
                         io.Combo.Input("model", options=["FP8", *PUBLIC_MODELS], default="FP8", optional=True,
                             tooltip="FP8 使用上方目录。GGUF 由同机 8195 服务加载。")],
                         outputs=[MODEL.Output("teacher"),
                                  io.String.Output("revision")])

    @classmethod
    def fingerprint_inputs(cls, directory="", use_fla=False, model="FP8"):
        if model in PUBLIC_MODELS:
            return (model, installed_model(model))
        path = locate_teacher(directory)
        stamps = []
        for name in MANIFEST["files"]:
            try:
                stat = (path / name).stat()
                stamps.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
            except FileNotFoundError:
                stamps.append(None)
        return (str(path), stamps, bool(list(path.glob("model-*-of-00008.safetensors"))), use_fla)

    @classmethod
    def execute(cls, directory="", use_fla=False, model="FP8"):
        if model in PUBLIC_MODELS:
            digest = installed_model(model)
            return io.NodeOutput(QuantizedModel(model, digest), digest)
        if model != "FP8":
            raise ValueError(f"Unknown Drama 3 model: {model}")
        path = validate_teacher(locate_teacher(directory))
        return io.NodeOutput(FlaTeacherPath(path) if use_fla else str(path), REVISION)


class Sampling(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3Sampling", display_name="Drama 3 · 采样设置",
                         category=CATEGORY, inputs=[
                             io.Float.Input("temperature", default=0.7, min=0.05, max=2.0, step=0.05),
                             io.Float.Input("top_p", default=0.9, min=0.05, max=1.0, step=0.01),
                             io.Int.Input("top_k", default=0, min=0, max=4097,
                                          tooltip="0 表示使用全部合法语音候选。"),
                             io.Int.Input("max_frames", default=1024, min=16, max=8192),
                         ], outputs=[SETTINGS.Output("settings")])

    @classmethod
    def execute(cls, temperature, top_p, top_k, max_frames):
        return io.NodeOutput(dict(temperature=temperature, top_p=top_p, top_k=top_k, max_frames=max_frames))


def _references(cast):
    refs = []
    for role in cast or ():
        if role.get("audio") is not None:
            refs.append(dict(speaker=role["speaker"], text=role["transcript"], audio=role["audio"]))
    return refs


def _preview_events(enabled):
    if not enabled:
        return None, lambda: None
    from server import PromptServer
    server = PromptServer.instance
    client = getattr(server, "client_id", None)
    if not client:
        return None, lambda: None
    token = uuid.uuid4().hex

    def send(chunk):
        wave = chunk["waveform"].detach().cpu().reshape(-1).clamp(-1, 1)
        pcm = (wave * 32767).round().to(torch.int16).numpy().astype("<i2", copy=False)
        server.send_sync("jr.drama3.audio", {"id": token, "rate": int(chunk["sample_rate"]),
                         "pcm": base64.b64encode(pcm.tobytes()).decode("ascii")}, sid=client)

    def end():
        server.send_sync("jr.drama3.audio", {"id": token, "end": True}, sid=client)
    return send, end


def render(teacher, text, seed, *, mode="line", direction="", cast=None, previous=None,
           infill=None, settings=None, live=False, on_audio_chunk=None):
    if not text.strip():
        raise ValueError("请输入台词或场景。")
    check = model_management.throw_exception_if_processing_interrupted
    check()
    options = {**DEFAULTS, **(settings or {})}
    bar = ProgressBar(options["max_frames"])
    start = time.monotonic()
    playback, finish = _preview_events(live)
    stream = playback or on_audio_chunk
    if playback is not None and on_audio_chunk is not None:
        def stream(chunk):
            on_audio_chunk(chunk)
            playback(chunk)
    if isinstance(teacher, QuantizedModel):
        try:
            return render_quantized(teacher, text, seed, mode, direction, cast, options, stream, check, previous)
        except IncompleteSpeechError as exc:
            raise IncompleteTakeError(str(exc)) from exc
        finally:
            finish()
    from .jr_runtime import UnfinishedSpeechError, synthesize
    try:
        result = synthesize(teacher, text, _references(cast), seed, direction=direction,
                            use_fla=isinstance(teacher, FlaTeacherPath),
                            mode=mode, previous=previous, infill=infill, **options,
                            progress=bar.update_absolute, stream=stream, cancel=check)
    except UnfinishedSpeechError as exc:
        raise IncompleteTakeError(str(exc)) from exc
    finally:
        finish()
    check()
    require_complete({"finish_reason": result["finish_reason"], "mode": mode})
    audio = result["audio"]
    wave = audio["waveform"]
    if wave.ndim != 3 or wave.shape[:2] != (1, 1) or wave.shape[-1] < 1 or audio["sample_rate"] != 44100:
        raise RuntimeError("教师返回的音频格式有误。")
    if not torch.isfinite(wave).all() or wave.abs().max().item() < 1.e-5:
        raise EmptyTakeError("教师返回静音或非有限采样值。请调整参考音、提示或种子重试。")
    elapsed = time.monotonic() - start
    length = wave.shape[-1] / 44100
    report = dict(revision=REVISION, mode=mode, seed=seed, sampling=options,
                  attention_backend="fla" if isinstance(teacher, FlaTeacherPath) else "torch",
                  frames=result["frames"], finish_reason=result["finish_reason"],
                  seconds=round(length, 4), elapsed_seconds=round(elapsed, 3),
                  real_time_factor=round(elapsed / length, 4))
    for field in ("segments", "edit", "host_expert_layers", "scene_corrections"):
        if result.get(field) is not None:
            report[field] = result[field]
    return audio, result["history"], report


class Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3Generate", display_name="Drama 3 · 本地语音生成",
                         category=CATEGORY, inputs=[
                             MODEL.Input("teacher"), io.String.Input("text", multiline=True),
                             io.Combo.Input("mode", options=["line", "design", "scene", "continue", "revise"],
                                 tooltip="line、design 支持全部模型。scene 使用 FP8 的整场提示。"
                                         "continue、revise 可接 FP8、五语或 NVFP4 的同模型 history。"),
                             io.String.Input("direction", default="", multiline=True,
                                 tooltip="导演意见或声音描述。场景 JSON 接入 text，这里可填写附加导演意见。"),
                             seed_input(), io.Boolean.Input("live", default=False),
                             CAST.Input("cast", optional=True), HISTORY.Input("previous", optional=True),
                             SETTINGS.Input("settings", optional=True),
                         ], outputs=[io.Audio.Output("audio"), HISTORY.Output("history"), io.String.Output("report")])

    @classmethod
    def execute(cls, teacher, text, mode, direction, seed, live, cast=None, previous=None, settings=None):
        if isinstance(teacher, QuantizedModel) and teacher.id == "drama-3-q4" and mode in ("continue", "revise"):
            raise ValueError("Q4 无法跨请求载入声学历史。请选 line，或改用五语、NVFP4、FP8。")
        if mode in ("continue", "revise") and previous is None:
            raise ValueError("续写和返修需要连接上一版 history。")
        if mode == "design" and not direction.strip():
            raise ValueError("请在 direction 输入要设计的声音特征。")
        audio, history, report = render(teacher, text, seed, mode=mode, direction=direction,
                                        cast=cast, previous=previous, settings=settings, live=live)
        return io.NodeOutput(audio, history, json.dumps(report, ensure_ascii=False))


class Infill(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3Infill", display_name="Drama 3 · 上下文局部补录",
                         description="用 FP8 的双侧上下文补录选区。此模式需要原始教师权重。",
                         category=CATEGORY, inputs=[
                             MODEL.Input("teacher"), io.Audio.Input("source"),
                             io.String.Input("revised_transcript", multiline=True,
                                             tooltip="修改后的完整原稿，包括选区前后文。"),
                             io.Float.Input("start_seconds", default=0, min=0, max=86400, step=0.01),
                             io.Float.Input("end_seconds", default=1, min=0, max=86400, step=0.01),
                             seed_input(), SETTINGS.Input("settings", optional=True),
                         ], outputs=[io.Audio.Output("audio"), HISTORY.Output("history"), io.String.Output("report")])

    @classmethod
    def execute(cls, teacher, source, revised_transcript, start_seconds, end_seconds, seed, settings=None):
        duration = source["waveform"].shape[-1] / source["sample_rate"]
        if not 0 <= start_seconds <= end_seconds <= duration:
            raise ValueError(f"选区必须位于 0 到 {duration:.3f} 秒之间。")
        region = dict(audio=source, text=revised_transcript, start_seconds=start_seconds, end_seconds=end_seconds)
        audio, history, report = render(teacher, revised_transcript, seed, mode="infill", infill=region, settings=settings)
        return io.NodeOutput(audio, history, json.dumps(report, ensure_ascii=False))


class Unload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3Unload", display_name="Drama 3 · 释放语音模型",
                         category=CATEGORY, is_output_node=True, inputs=[io.Audio.Input("after_audio", optional=True),
                         MODEL.Input("teacher", optional=True, tooltip="留空释放 FP8。连接量化模型只卸载该引擎，网关保留。")],
                         outputs=[io.String.Output("report"), io.Audio.Output("audio")])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, after_audio=None, teacher=None):
        if isinstance(teacher, QuantizedModel):
            unload_model(teacher.id)
            return io.NodeOutput(f"已释放 {teacher.id}，下次生成时重新加载。", after_audio)
        from .jr_runtime import unload
        unload()
        return io.NodeOutput("已释放本插件的教师缓存", after_audio)


def _take_dir():
    from pathlib import Path
    from folder_paths import get_output_directory
    return Path(get_output_directory()) / "JR-Drama3" / "takes"


def _cached_render(folder, identity, teacher, text, seed, cast, settings, **kwargs):
    if isinstance(teacher, QuantizedModel):
        identity = {**identity, "model": teacher.id, "revision": teacher.sha256}
        if any(role.get("audio") is not None for role in cast or ()):
            identity["reference_echo_check"] = 1
    if isinstance(teacher, FlaTeacherPath):
        identity = {**identity, "attention_backend": "fla-0.6.0"}
    key = key_for(identity)
    lookup = [key]
    # Batch v1 omitted direction and inserted source=None on independent
    # lines. Keep all previously locked IDs and linked chains usable.
    if identity["mode"] == "line" and not identity.get("direction") and not identity.get("source"):
        canonical = {**identity, "direction": ""}
        canonical.pop("source", None)
        legacy = {**canonical, "source": None}
        legacy.pop("direction")
        key = key_for(canonical)
        lookup = list(dict.fromkeys([lookup[0], key, key_for(legacy)]))
    for existing in lookup:
        result = read_take(folder, existing)
        if result is not None:
            return existing, result
    result = render(teacher, text, seed, cast=cast, settings=settings, **kwargs)
    save_take(folder, key, result)
    return key, result


class Candidates(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3Candidates", display_name="Drama 3 · 生成候选版本",
                         category=CATEGORY, inputs=[MODEL.Input("teacher"), io.String.Input("text", multiline=True),
                         seed_input(), io.Int.Input("count", min=1, max=16, default=3),
                         io.String.Input("direction", default=""), CAST.Input("cast", optional=True),
                         SETTINGS.Input("settings", optional=True),
                         io.Combo.Input("mode", options=["line", "design"], default="line", optional=True)],
                         outputs=[TAKES.Output("takes"), io.String.Output("report")])

    @classmethod
    def execute(cls, teacher, text, seed, count, direction, cast=None, settings=None, mode="line"):
        if mode not in ("line", "design"):
            raise ValueError("候选模式必须为 line 或 design。")
        if mode == "design" and not direction.strip():
            raise ValueError("请在 direction 输入要设计的声音特征。")
        settings = {**DEFAULTS, **(settings or {})}
        base = {"format": 1, "revision": REVISION, "text": text, "cast": cast_key(cast),
                "mode": mode, "direction": direction, "settings": settings}
        keys = []
        reports = []
        for i in range(count):
            model_management.throw_exception_if_processing_interrupted()
            sample_seed = (seed + i) & MAX_SEED
            try:
                key, (_, _, report) = _cached_render(_take_dir(), dict(base, seed=sample_seed), teacher,
                                          text, sample_seed, cast, settings, mode=mode, direction=direction)
            except IncompleteTakeError as exc:
                keys.append(None)
                reports.append(dict(candidate=i+1, seed=sample_seed, error=str(exc)))
                continue
            keys.append(key)
            reports.append(dict(candidate=i+1, take=key, **report))
        if not any(keys):
            raise IncompleteTakeError(f"全部 {len(keys)} 个候选均未生成完整语音。{reports[-1]['error']}")
        return io.NodeOutput(keys, json.dumps(reports, ensure_ascii=False))


class SelectTake(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3SelectTake", display_name="Drama 3 · 选择或锁定候选",
                         category=CATEGORY, inputs=[io.Int.Input("number", default=1, min=1, max=16),
                         io.String.Input("locked_id", default="", tooltip="保存报告内的 take ID 即可固定选择。"),
                         TAKES.Input("takes", optional=True)], outputs=[io.Audio.Output("audio"), HISTORY.Output("history"),
                         io.String.Output("locked_id")])

    @classmethod
    def execute(cls, number, locked_id="", takes=None):
        import re
        if locked_id.strip():
            key = locked_id.strip()
        elif takes and 1 <= number <= len(takes):
            key = takes[number - 1]
            if key is None:
                raise IncompleteTakeError(f"第 {number} 个候选未完整生成。请从报告中选择成功的编号。")
        else:
            raise ValueError("请选择有效候选或输入已保存的 take ID。")
        if not re.fullmatch("[0-9a-f]{64}", key):
            raise ValueError("take ID 无效。")
        result = read_take(_take_dir(), key)
        if result is None:
            raise FileNotFoundError(f"未找到已完成的本地 take：{key}")
        return io.NodeOutput(result[0], result[1], key)


class Batch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3Batch", display_name="Drama 3 · 逐句生成与断点续作",
                         category=CATEGORY, inputs=[MODEL.Input("teacher"), CUES.Input("cues"), seed_input(),
                         io.Combo.Input("continuity", options=["auto", "linked", "independent"], default="auto",
                             tooltip="auto 对 Q4 逐句独立，对其他模型使用声学历史。"),
                         CAST.Input("cast", optional=True), SETTINGS.Input("settings", optional=True)],
                         outputs=[io.Audio.Output("clips", is_output_list=True),
                         HISTORY.Output("last_history"), io.String.Output("report")])

    @classmethod
    def execute(cls, teacher, cues, seed, continuity, cast=None, settings=None):
        return io.NodeOutput(*render_cues(teacher, cues, seed, continuity, cast, settings))


def render_cues(teacher, cues, seed, continuity, cast, settings):
    from .jr_script_core import SPEAKER, checked_cast, normalize_name
    if not cues:
        raise ValueError("台本中没有台词。")
    if continuity == "auto":
        continuity = "independent" if isinstance(teacher, QuantizedModel) and teacher.id == "drama-3-q4" else "linked"
    if continuity not in ("linked", "independent"):
        raise ValueError("请选择 linked 或 independent 连续性模式。")
    if isinstance(teacher, QuantizedModel) and teacher.id == "drama-3-q4" and continuity == "linked":
        raise ValueError("Q4 批量与长文请选 independent，各句复用角色参考。")
    settings = {**DEFAULTS, **(settings or {})}
    cast = checked_cast(cast)
    voice = cast_key(cast)
    names = {normalize_name(r["name"]): r["speaker"] for r in cast}
    occupied = set(names.values())
    for cue in cues:
        occupied.update(int(marker) for marker in SPEAKER.findall(cue["text"]))
        leading = SPEAKER.match(cue["text"].lstrip())
        if leading and cue.get("role"):
            name = normalize_name(cue["role"])
            speaker = int(leading[1])
            if names.setdefault(name, speaker) != speaker:
                raise ValueError(f"角色 {cue['role']} 的文字编号与角色绑定不一致。")
    clips, summary = [], []
    previous = None
    prior_key = None
    for i, cue in enumerate(cues):
        model_management.throw_exception_if_processing_interrupted()
        text = cue["text"].strip()
        role = normalize_name(cue.get("role") or "")
        if role and not text.startswith("<|speaker:"):
            if role not in names:
                available = next((n for n in range(16) if n not in occupied), None)
                if available is None:
                    raise ValueError("角色数量超过模型的 16 个编号。")
                names[role] = available
                occupied.add(available)
            text = f"<|speaker:{names[role]}|>" + text
        source = prior_key if continuity == "linked" else None
        mode = "continue" if source else "line"
        sample_seed = (seed + i) & MAX_SEED
        identity = dict(format=1, revision=REVISION, text=text, cast=voice, mode=mode,
                        settings=settings, seed=sample_seed, source=source)
        prior_key, (audio, previous, report) = _cached_render(_take_dir(), identity, teacher,
            text, sample_seed, cast if not source or isinstance(teacher, QuantizedModel) else None, settings, mode=mode,
            previous=previous if source else None)
        clips.append(audio)
        summary.append(dict(cue=cue.get("id", i+1), take=prior_key, **report))
    return clips, previous, json.dumps(summary, ensure_ascii=False)


class LongForm(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3LongForm", display_name="Drama 3 · 连续长文",
                         category=CATEGORY, inputs=[MODEL.Input("teacher"),
                         io.String.Input("text", multiline=True), seed_input(),
                         io.Int.Input("max_bytes", min=32, max=4096, default=300),
                         io.Float.Input("gap_seconds", min=0, max=5, default=0, step=0.01),
                         CAST.Input("cast", optional=True), SETTINGS.Input("settings", optional=True),
                         io.Combo.Input("continuity", options=["auto", "linked", "independent"], default="auto", optional=True,
                             tooltip="auto 对 Q4 逐段独立，对其他模型使用声学历史。")],
                         outputs=[io.Audio.Output("audio"), io.Audio.Output("clips", is_output_list=True),
                         HISTORY.Output("history"), io.String.Output("report")])

    @classmethod
    def execute(cls, teacher, text, seed, max_bytes, gap_seconds, cast=None, settings=None, continuity="auto"):
        from .jr_script_core import chunk_script
        chunks, warnings = chunk_script(text, max_bytes)
        clips, history, report = render_cues(teacher, [{"text": s} for s in chunks],
                                             seed, continuity, cast, settings)
        gap = clips[0]["waveform"].new_zeros(1, 1, round(44100 * gap_seconds))
        pieces = []
        for audio in clips:
            if pieces and gap.numel():
                pieces.append(gap)
            pieces.append(audio["waveform"])
        full = {"waveform": torch.cat(pieces, dim=-1), "sample_rate": 44100}
        record = {"chunks": json.loads(report), "warnings": warnings}
        return io.NodeOutput(full, clips, history, json.dumps(record, ensure_ascii=False))


class CheckedTake(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3CheckedTake", display_name="Drama 3 · 回听校验与限次重录",
                         category=CATEGORY, inputs=[MODEL.Input("teacher"), io.String.Input("text", multiline=True),
                         io.String.Input("spoken_transcript", default="", multiline=True,
                             tooltip="预期朗读的纯文本；留空则使用台词。"),
                         io.Combo.Input("language", options=["auto", "zh", "en", "ja", "ko"]), seed_input(),
                         io.Int.Input("retry_limit", min=0, max=5, default=1),
                         io.Float.Input("max_error", min=0, max=1, default=0.2, step=0.01),
                         CAST.Input("cast", optional=True), SETTINGS.Input("settings", optional=True)],
                         outputs=[io.Audio.Output("audio"), HISTORY.Output("history"), TAKES.Output("takes"),
                                  io.String.Output("quality")])

    @classmethod
    def execute(cls, teacher, text, spoken_transcript, language, seed, retry_limit, max_error, cast=None, settings=None):
        from .jr_analysis_asr import transcribe
        from .jr_analysis_quality import review
        expected = spoken_transcript.strip() or text
        settings = {**DEFAULTS, **(settings or {})}
        voice = cast_key(cast)
        keys, reviews = [], []
        best, best_score = None, None
        for i in range(retry_limit + 1):
            model_management.throw_exception_if_processing_interrupted()
            sample_seed = (seed + i) & MAX_SEED
            identity = dict(format=1, revision=REVISION, text=text, cast=voice, mode="line", direction="",
                            settings=settings, seed=sample_seed)
            try:
                key, (audio, history, info) = _cached_render(_take_dir(), identity, teacher, text,
                                                              sample_seed, cast, settings)
            except IncompleteTakeError as exc:
                reviews.append({"seed": sample_seed, "error": str(exc)})
                continue
            keys.append(key)
            hearing = transcribe(audio, "Whisper small", device="cpu", language=language)
            quality = review(audio, expected, hearing["text"], max_error=max_error, attempt=i, retry_limit=retry_limit)
            hard_flags = set(quality["flags"]) - {"review_long_silence"}
            score = (len(hard_flags), quality.get("wer", quality["cer"]))
            reviews.append(dict(take=key, seed=sample_seed, heard=hearing["text"], finish=info["finish_reason"], **quality))
            if best_score is None or score < best_score:
                best, best_score = (audio, history, key), score
            if not hard_flags:
                break
        if best is None:
            raise EmptyTakeError(f"已尝试 {len(reviews)} 次，均未得到完整音频。{reviews[-1]['error']}")
        return io.NodeOutput(best[0], best[1], keys, json.dumps({"selected": best[2], "attempts": reviews},
                                                                ensure_ascii=False))


NODES = [Teacher, Sampling, Generate, Infill, Candidates, SelectTake, CheckedTake, Batch, LongForm, Unload]
