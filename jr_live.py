"""Canvas text input for an already queued local execution."""
from __future__ import annotations

import json
import queue
import secrets
import threading
import time

from aiohttp import web
from comfy_api.latest import io
from comfy import model_management
from comfy.utils import ProgressBar
from .jr_generate import CATEGORY, MODEL, CAST, SETTINGS, HISTORY, DEFAULTS, seed_input, _preview_events, _references
from .jr_teacher import FlaTeacherPath
from .jr_gguf import QuantizedModel, render_line

_active = {}
_lock = threading.Lock()


def register_canvas_control():
    from server import PromptServer
    server = PromptServer.instance
    if getattr(server, "_jr_drama3_canvas", False):
        return

    @server.routes.post("/jr-drama3/canvas/text")
    async def receive(request):
        if request.content_length is None or request.content_length > 32768:
            raise web.HTTPBadRequest(text="Invalid input size")
        try:
            body = await request.json()
            token, action = body["session"], body["action"]
            if not isinstance(token, str) or action not in ("append", "finish"):
                raise ValueError
            text = body.get("text", "")
            if action == "append" and (not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000):
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise web.HTTPBadRequest(text="Invalid canvas input")
        with _lock:
            session = _active.get(token)
            if session is None or session["closed"]:
                raise web.HTTPGone(text="This canvas session has ended")
            try:
                session["queue"].put_nowait(text.strip() if action == "append" else None)
            except queue.Full:
                raise web.HTTPTooManyRequests(text="Wait for the current lines to finish")
            if action == "finish":
                session["closed"] = True
        return web.json_response({"ok": True})

    server._jr_drama3_canvas = True


class LiveText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="JRDrama3LiveText", display_name="Drama 3 · 实时追加台词",
                         category=CATEGORY, is_output_node=True,
                         description="画布可追加或结束台词。FP8 预读下一段。五语、NVFP4 回传上一段音频。Q4 逐段使用角色参考。",
                         hidden=[io.Hidden.unique_id],
                         inputs=[MODEL.Input("teacher"), io.String.Input("first_line", multiline=True),
                                 seed_input(), io.Boolean.Input("live", default=True),
                                 io.Int.Input("idle_seconds", default=180, min=15, max=3600),
                                 CAST.Input("cast", optional=True), SETTINGS.Input("settings", optional=True)],
                         outputs=[io.Audio.Output("audio"), HISTORY.Output("history"), io.String.Output("report")])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, teacher, first_line, seed, live, idle_seconds, cast=None, settings=None):
        from server import PromptServer
        from comfy_api.latest import ui
        if not first_line.strip():
            raise ValueError("请先填写第一段台词。")
        server = PromptServer.instance
        client = server.client_id
        if not client:
            raise ValueError("实时追加台词需要在本地 ComfyUI 画布提交。")
        token = secrets.token_urlsafe(32)
        inbox = queue.Queue(maxsize=64)
        check = model_management.throw_exception_if_processing_interrupted
        event = {"node": str(cls.hidden.unique_id), "session": token}

        def chunks():
            yield first_line.strip()
            while True:
                start = time.monotonic()
                while True:
                    check()
                    if time.monotonic() - start >= idle_seconds:
                        return
                    try:
                        line = inbox.get(timeout=0.25)
                        break
                    except queue.Empty:
                        continue
                if line is None:
                    return
                yield line

        with _lock:
            _active[token] = {"queue": inbox, "closed": False}
        opts = {**DEFAULTS, **(settings or {})}
        progress = ProgressBar(opts["max_frames"])
        stream, finish = _preview_events(live)
        try:
            server.send_sync("jr.drama3.text", {**event, "active": True}, sid=client)
            if isinstance(teacher, QuantizedModel):
                import torch
                from .jr_realtime_client import RATE
                clips, segments = [], []
                cursor = 0
                linked = teacher.id != "drama-3-q4"
                history = None
                for i, line in enumerate(chunks()):
                    mode = "continue" if i and linked else "line"
                    audio, history, info = render_line(teacher, line, (seed + i) % 2**32, mode, "",
                                                       cast, opts, stream, check, history)
                    wave = audio["waveform"]
                    clips.append(wave)
                    segments.append({"text": line, "start_sample": cursor, "end_sample": cursor + wave.shape[-1],
                                     "finish_reason": info["finish_reason"]})
                    cursor += wave.shape[-1]
                audio = {"waveform": torch.cat(clips, dim=-1), "sample_rate": RATE}
                report = {"model": teacher.id, "seed": seed, "segments": segments,
                          "continuity": "previous_take" if linked else "independent"}
                return io.NodeOutput(audio, history, json.dumps(report, ensure_ascii=False), ui=ui.PreviewAudio(audio, cls=cls))
            from .jr_runtime import synthesize_stream
            result = synthesize_stream(teacher, chunks(), _references(cast), seed, **opts,
                        use_fla=isinstance(teacher, FlaTeacherPath),
                        progress=progress.update_absolute, stream=stream, cancel=check)
        finally:
            with _lock:
                _active.pop(token, None)
            finish()
            server.send_sync("jr.drama3.text", {**event, "active": False}, sid=client)
        report = {k: result[k] for k in ("frames", "seed", "segments", "finish_reason")}
        report["attention_backend"] = "fla" if isinstance(teacher, FlaTeacherPath) else "torch"
        return io.NodeOutput(result["audio"], result["history"], json.dumps(report, ensure_ascii=False),
                             ui=ui.PreviewAudio(result["audio"], cls=cls))
