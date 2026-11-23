"""Speech request validation, local voice catalog and ComfyUI graphs."""
import json
import math
from pathlib import Path
import re
import sqlite3
from contextlib import closing

from jr_script_core import chunk_script


MODEL = "drama-3"
VOICES = {
    "default": "A neutral adult narrator with a natural conversational voice.",
    "alloy": "An even, warm midrange voice with clear, balanced articulation.",
    "ash": "A low, lightly gravelly voice, close to the microphone and understated.",
    "ballad": "A gentle lyrical storyteller with long, smoothly connected vowels.",
    "coral": "A bright, sociable woman with lively pitch changes and crisp consonants.",
    "echo": "A composed male lecturer with an even rhythm and resonant lower register.",
    "fable": "An expressive mature storyteller with a textured, friendly voice.",
    "onyx": "A deep, steady baritone with rounded resonance and deliberate phrasing.",
    "nova": "An energetic young woman with a light, forward voice and nimble diction.",
    "sage": "A calm, reflective mature woman with a soft lower register.",
    "shimmer": "An airy soprano with a light breath and a gentle rising lilt.",
    "verse": "A youthful tenor with relaxed rhythm and an easy conversational tone.",
    "marin": "A poised contralto with precise diction and measured pacing.",
    "cedar": "A warm mature bass with full chest resonance and unhurried phrasing.",
}
FORMATS = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac",
           "flac": "audio/flac", "wav": "audio/wav", "pcm": "application/octet-stream"}


class ApiError(Exception):
    def __init__(self, status, message, *, param=None, code=None):
        super().__init__(message)
        self.status, self.param, self.code = status, param, code


def catalog(db, voice_id=None):
    path = Path(db)
    if not path.is_file():
        return []
    select = "SELECT voice_id AS id, name, language, description, tags FROM voices"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as con:
        con.row_factory = sqlite3.Row
        rows = (con.execute(select + " WHERE voice_id=?", (voice_id,)) if voice_id else
                con.execute(select + " ORDER BY row_id DESC"))
        return [{"object": "voice", "source": "library", **dict(row)} for row in rows]


def speech_params(body):
    if not isinstance(body, dict):
        raise ApiError(400, "Expected a JSON object.")
    allowed = {"model", "input", "voice", "instructions", "speed", "seed", "response_format", "stream_format"}
    for field in body:
        if field not in allowed:
            raise ApiError(400, f"Unsupported speech parameter: {field}", param=field)
    if body.get("model") != MODEL:
        raise ApiError(400, "Select the local drama-3 model.", param="model", code="model_not_found")
    text = body.get("input")
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        raise ApiError(400, "input must contain 1–4096 characters.", param="input")
    voice = body.get("voice", "default")
    if isinstance(voice, dict):
        voice = voice.get("id")
    if not isinstance(voice, str) or (voice not in VOICES and not re.fullmatch(r"[a-f0-9]{64}", voice)):
        raise ApiError(400, "Unknown voice. See /v1/audio/voices.", param="voice")
    direction = body.get("instructions", "")
    if not isinstance(direction, str) or len(direction) > 4096:
        raise ApiError(400, "instructions must contain at most 4096 characters.", param="instructions")
    speed = body.get("speed", 1.)
    if type(speed) not in (float, int) or not .25 <= speed <= 4:
        raise ApiError(400, "speed must be a number between 0.25 and 4.0.", param="speed")
    seed = body.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ApiError(400, "seed must be an integer between 0 and 4294967295.", param="seed")
    fmt = body.get("response_format", "mp3")
    if not isinstance(fmt, str) or fmt not in FORMATS:
        raise ApiError(400, "Unsupported response format.", param="response_format")
    if body.get("stream_format", "audio") != "audio":
        raise ApiError(400, "This provider returns completed audio bytes.", param="stream_format")
    return dict(text=text.strip(), voice=voice, direction=direction.strip(), speed=speed, seed=seed, fmt=fmt)


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def build_graph(request, unload=False):
    text, voice = request["text"], request["voice"]
    try:
        pieces, _ = chunk_script(text, 300)
    except ValueError as exc:
        raise ApiError(400, str(exc), param="input") from exc
    # Pack short sentences before building the graph. Maintain scope/phoneme
    # pairs and the existing role bindings from the node's own script parser.
    chunks = []
    for piece in pieces:
        if chunks and len((chunks[-1] + " " + piece).encode("utf-8")) <= 300:
            chunks[-1] += " " + piece
        else:
            chunks.append(piece)
    if len(text.encode("utf-8")) <= 300:
        chunks = [text]
    graph = {"teacher": node("JRDrama3Teacher", directory="")}
    is_library = voice not in VOICES
    if is_library:
        graph["find"] = node("JRDrama3FindVoice", query=voice, language="", rank=1)
        graph["voice_id"] = node("PreviewAny", source=["find", 2])
        graph["cast"] = node("JRDrama3Reference", name="Narrator", speaker=0, transcript=["find", 1],
                             audio=["find", 0], description="", language="")
    reports = []
    for i, chunk in enumerate(chunks):
        take = f"take_{i}"
        report = f"report_{i}"
        direction = request["direction"]
        mode = "continue" if i else ("line" if is_library else "design")
        extra = {}
        if i:
            extra["previous"] = [f"take_{i-1}", 1]
        elif is_library:
            extra["cast"] = ["cast", 0]
        else:
            direction = VOICES[voice] + (f" Performance direction: {direction}" if direction else "")
        graph[take] = node("JRDrama3Generate", teacher=["teacher", 0], text=chunk, mode=mode,
                           direction=direction, live=False, seed=(request["seed"] + i) % 2**32, **extra)
        graph[report] = node("PreviewAny", source=[take, 2])
        reports.append(report)
        if i:
            join = f"join_{i}"
            graph[join] = node("JRDrama3JoinAudio", first=audio, second=[take, 0], gap_s=0., crossfade_s=0.)
            audio = [join, 0]
        else:
            audio = [take, 0]
    if unload:
        graph["unload"] = node("JRDrama3Unload", after_audio=audio)
        audio = ["unload", 1]
    graph["audio"] = node("PreviewAudio", audio=audio)
    return graph, reports


def completed_audio(entry, reports, voice):
    status = entry.get("status", {})
    if status.get("status_str") != "success":
        for kind, item in status.get("messages", []):
            if kind == "execution_error":
                raise ApiError(502, str(item.get("exception_message") or "ComfyUI generation failed.").strip()[:1000])
            if kind == "execution_interrupted":
                raise ApiError(409, "The ComfyUI job was interrupted.", code="job_interrupted")
        raise ApiError(502, "ComfyUI did not finish the speech job.")
    try:
        outputs = entry["outputs"]
        if voice not in VOICES and outputs["voice_id"]["text"][0] != voice:
            raise ApiError(502, "ComfyUI returned a different saved voice ID.")
        for report in reports:
            item = json.loads(outputs[report]["text"][0])
            seconds = float(item["seconds"])
            if item["finish_reason"] != "eos" or not math.isfinite(seconds) or seconds <= 0:
                raise ApiError(502, "The teacher did not finish the complete input.", code="incomplete_speech")
        info = outputs["audio"]["audio"][0]
        if info.get("type") != "temp" or not info["filename"].endswith(".flac"):
            raise ApiError(502, "ComfyUI returned an unexpected preview format.")
        return {name: info[name] for name in ("filename", "type", "subfolder") if name in info}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ApiError(502, "ComfyUI returned an incomplete speech report.") from exc
