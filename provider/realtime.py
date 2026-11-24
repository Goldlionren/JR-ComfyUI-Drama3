"""Local streaming API for the resident quantized engine. Built with Fish Audio."""

import argparse
import asyncio
import base64
from contextlib import aclosing
import io
import json
import secrets
import wave

import numpy as np
import regex
from aiohttp import ClientSession, ClientTimeout, web

from jr_realtime_client import PORT, PRESET_VOICES, RATE, pcm_event
from jr_realtime_paths import MODELS, PUBLIC_MODELS, engine_binary, model_file
from jr_script_core import MARKS, SPEAKER, apply_to_spoken, chunk_script
from .audio import encode_audio, local_ffmpeg
from .q4_voice import load_q4_voice
from .realtime_voices import register_voice, voice_options
from .server import LAN_INTERFACE, LAN_NETWORKS, api_boundary, default_db
from .speech import ApiError, FORMATS, catalog

ENGINE_PORT = 8196
MODEL = "drama-3"
SDK_PCM_RATE = 24000
# Match whole emoji clusters (including flags, skin tones and joined family emoji).
EMOJI = regex.compile(r"[#*0-9]\ufe0f?\u20e3|(?=[\p{Extended_Pictographic}\p{Regional_Indicator}\p{Emoji_Modifier}])\X")
SPOKEN_CHAR = regex.compile(r"(?!\p{Default_Ignorable_Code_Point})[\p{L}\p{N}]")


def directed_chunks(text, instructions):
    if not isinstance(instructions, str):
        raise ValueError("instructions must be a short plain-text acting direction.")
    direction = " ".join(instructions.split())
    if len(direction.encode()) > 200 or any(c in direction for c in "[]<>"):
        raise ValueError("Use up to 200 UTF-8 bytes of plain text in instructions. Put inline tags in input.")
    cue = f"[{direction}] " if direction else ""
    # Empty cue and phoneme pairs have no token payload for the script parser.
    text = text.replace("[]", " ").replace("<|phoneme_start|><|phoneme_end|>", " ")
    chunks, _ = chunk_script(text, 300 - len(cue.encode()), allow_empty=True)
    output = []
    for chunk in chunks:
        if EMOJI.search(chunk):
            chunk = apply_to_spoken(chunk, lambda part: EMOJI.sub(" ", part)).strip()
        speaker = SPEAKER.match(chunk)
        offset = speaker.end() if speaker else 0
        line = chunk[offset:].lstrip()
        # Directions alone can make the model recite the reference. Phonetic
        # spellings are dialogue, so keep their payload when inspecting the text.
        spoken = MARKS.sub("", line.replace("<|phoneme_start|>", "").replace("<|phoneme_end|>", ""))
        if SPOKEN_CHAR.search(spoken):
            output.append((chunk[:offset] + " " if speaker else "") + cue + line)
    return output


async def silent_pcm(rate):
    # A decorations-only request still needs valid audio for WAV previews and SSE clients.
    yield bytes(2 * (rate // 50))


async def sdk_pcm(source):
    """Resample one continuous request, including any native sentence boundaries."""
    import soxr

    resampler = soxr.ResampleStream(RATE, SDK_PCM_RATE, 1, dtype="int16")
    async with aclosing(source):
        async for pcm in source:
            samples = resampler.resample_chunk(np.frombuffer(pcm, dtype="<i2"))
            if samples.size:
                yield samples.astype("<i2", copy=False).tobytes()
        tail = resampler.resample_chunk(np.empty(0, dtype=np.int16), last=True)
        if tail.size:
            yield tail.astype("<i2", copy=False).tobytes()


def create_app(backend, engine_port=ENGINE_PORT, *, voice_db=None, lan=None, extra_networks=(), switch=None):
    app = web.Application(middlewares=[api_boundary], client_max_size=7 * 1024 * 1024)
    if lan is not None:
        app[LAN_INTERFACE] = lan
        app[LAN_NETWORKS] = (lan.network, *extra_networks)
    base = f"http://127.0.0.1:{engine_port}"
    db = voice_db if voice_db is not None else default_db()
    engine = None
    busy = False

    async def connection(app):
        nonlocal engine
        async with ClientSession(
            timeout=ClientTimeout(total=180), read_bufsize=4 * 1024 * 1024, trust_env=False
        ) as engine:
            yield

    app.cleanup_ctx.append(connection)

    def catalog_entry(id_, selected):
        return {"id": id_, "object": "model", "owned_by": "jr-drama3", "backend": selected,
                "active": selected == backend,
                "installed": selected == backend or (engine_binary(selected).is_file() and model_file(selected).is_file())}

    async def models(request):
        entries = [catalog_entry(MODEL, backend)]
        entries.extend(catalog_entry(name, key) for name, key in PUBLIC_MODELS.items() if switch or key == backend)
        return web.json_response({"object": "list", "data": entries})

    async def health(request):
        async with engine.get(base + "/v1/models", allow_redirects=False) as r:
            if r.status != 200:
                raise ApiError(503, "Local quantized engine unavailable.")
            items = (await r.json())["data"]
            if not any(m["id"] == MODEL for m in items):
                raise ApiError(503, "Expected Drama 3 engine on the configured port.")
        return web.json_response({"object": "list", "data": [catalog_entry(MODEL, backend)]})

    def requested_backend(body):
        name = body.get("model", MODEL)
        if name == MODEL:
            return backend
        if not isinstance(name, str) or name not in PUBLIC_MODELS:
            raise ApiError(400, "Unknown model. See /v1/models.")
        target = PUBLIC_MODELS[name]
        if target != backend and switch is None:
            raise ApiError(503, "Start the managed realtime service to switch models.")
        return target

    async def activate(target):
        nonlocal backend
        if switch is None:
            return

        async def change():
            nonlocal backend
            try:
                await asyncio.to_thread(switch, target)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ApiError(503, f"Could not load {target}: {exc}") from exc
            backend = target

        task = asyncio.create_task(change())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # The native process transition outlives the HTTP client; keep the engine reserved.
            await task
            raise

    async def select_model(request):
        nonlocal busy
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise ApiError(400, "Invalid selection JSON.")
        if not isinstance(body, dict) or set(body) != {"model"}:
            raise ApiError(400, "Supply a model ID from /v1/models.")
        target = requested_backend(body)
        if busy:
            raise ApiError(429, "Speech engine busy.")
        busy = True
        try:
            await activate(target)
            return web.json_response(catalog_entry("drama-3-" + backend, backend))
        finally:
            busy = False

    async def list_voices(request):
        presets = PRESET_VOICES if backend == "5lang" else ("zh", "en")
        rows = [{"id": key, "object": "voice", "source": "preset"} for key in presets]
        rows.extend(await asyncio.to_thread(catalog, db))
        return web.json_response({"object": "list", "data": rows})

    async def save_voice(request):
        try:
            body = await request.json()
            entry = await asyncio.to_thread(register_voice, db, body)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc
        return web.json_response(entry)

    async def chunks(texts, voice, seed):
        if backend == "q4" and "voice_ref" in voice:
            voice = {"voice": await load_q4_voice(engine, base, voice["voice_ref"], voice["reference_text"])}
        for i, text in enumerate(texts):
            data = {
                "model": MODEL,
                "input": text,
                **voice,
                "response_format": "pcm",
                "stream_format": "sse",
            }
            part_seed = (seed + i) % 2**32
            if backend != "q4":
                # Keep each validated sentence intact. The engine's default
                # 128-codepoint split can otherwise create an unchecked tail.
                data["options"] = {"seed": part_seed, "text_chunk_size": max(300, len(text))}
            else:
                data["seed"] = part_seed
            async with engine.post(base + "/v1/audio/speech", json=data, allow_redirects=False) as r:
                if r.status != 200:
                    raise ApiError(502, (await r.content.read(1024)).decode(errors="replace"))
                count = 0
                async for line in r.content:
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        raise RuntimeError("Engine ended before confirming the sentence.")
                    event = json.loads(payload)
                    if event.get("type") == "error":
                        error = event.get("error", {})
                        message = str(error.get("message", error)) if isinstance(error, dict) else str(error)
                        if "leaves no room to generate" in message:
                            raise ApiError(400, "Reference audio, transcript and dialogue exceed the model context. "
                                           "Use a shorter excerpt with its matching transcript.")
                        raise ApiError(502, message[:1024] or "Native speech generation failed.")
                    pcm = pcm_event(event, count)
                    if pcm is None:
                        break
                    count += len(pcm) // 2
                    yield pcm
                else:
                    raise RuntimeError("Incomplete sentence stream.")

    async def speech(request):
        nonlocal busy
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise ApiError(400, "Invalid speech JSON.")
        if not isinstance(body, dict):
            raise ApiError(400, "Expected a drama-3 speech request.")
        target = requested_backend(body)
        if set(body) - {"model", "input", "seed", "voice", "response_format", "stream_format",
                        "instructions", "voice_ref", "reference_text"}:
            raise ApiError(400, "Unsupported realtime speech option.")
        text = body.get("input")
        if not isinstance(text, str) or len(text) > 4096:
            raise ApiError(400, "Supply an input string of at most 4096 characters.")
        seed = body.get("seed", secrets.randbits(32))
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ApiError(400, "Invalid seed.")
        fmt, stream = body.get("response_format", "pcm"), body.get("stream_format", "audio")
        if not isinstance(fmt, str) or fmt not in FORMATS:
            raise ApiError(400, f"Unsupported response_format {str(fmt)[:64]!r}. Choose {', '.join(FORMATS)}.",
                           param="response_format")
        if stream not in ("audio", "sse") or (stream == "sse" and fmt != "pcm"):
            raise ApiError(400, f"Unsupported stream_format {str(stream)[:64]!r} for {fmt}. "
                           "Use audio for files or sse with pcm.", param="stream_format")
        try:
            instructions = body.get("instructions", "")
            texts = directed_chunks(text, instructions)
            if texts:
                options = await asyncio.to_thread(voice_options, body, target, db)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc
        result = None
        # The OpenAI speech SDK explicitly requests octet-stream and plays raw PCM at 24 kHz.
        binary_pcm = fmt == "pcm" and stream == "audio" and (
            request.headers.get("Accept", "").strip().lower() == "application/octet-stream"
        )
        rate = SDK_PCM_RATE if binary_pcm else RATE
        if texts:
            if busy:
                raise ApiError(429, "Speech engine busy.")
            busy = True
            source = chunks(texts, options, seed)
            if binary_pcm:
                source = sdk_pcm(source)
        else:
            source = silent_pcm(rate)
        headers = {"X-Sample-Rate": str(rate), "X-Speech-Seed": str(seed),
                   "Cache-Control": "no-store", "Vary": "Accept"}
        try:
            if texts:
                await activate(target)
            if fmt != "pcm":
                executable = await asyncio.to_thread(local_ffmpeg) if fmt != "wav" else None
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as w:
                    w.setparams((1, 2, RATE, 0, "NONE", ""))
                    async for pcm in source:
                        w.writeframesraw(pcm)
                audio = buffer.getvalue()
                if executable:
                    audio = await encode_audio(audio, "wav", fmt, 1.0, executable)
                if fmt == "opus":
                    headers["X-Sample-Rate"] = "48000"
                return web.Response(body=audio, content_type=FORMATS[fmt], headers=headers)
            count = 0
            async for pcm in source:
                if result is None:
                    result = web.StreamResponse(headers=headers)
                    result.content_type = "application/octet-stream" if binary_pcm else (
                        "text/event-stream" if stream == "sse" else "audio/pcm"
                    )
                    await result.prepare(request)
                count += len(pcm) // 2
                if stream == "sse":
                    await result.write(
                        b'data: {"type":"speech.audio.delta","audio":"' + base64.b64encode(pcm) + b'"}\n\n'
                    )
                else:
                    await result.write(pcm)
            if stream == "sse":
                done = {"type": "speech.audio.done", "stop": "complete", "samples": count}
                await result.write(b"data: " + json.dumps(done).encode() + b"\n\n")
            await result.write_eof()
            return result
        except (Exception, asyncio.CancelledError):
            if result is not None and result.prepared and request.transport:
                request.transport.close()  # Signal truncation to raw-PCM clients as well as SSE clients.
            raise
        finally:
            await source.aclose()
            if texts:
                busy = False

    app.router.add_get("/", models)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/models/select", select_model)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/audio/voices", list_voices)
    app.router.add_post("/v1/audio/voices", save_voice)
    app.router.add_post("/v1/audio/speech", speech)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    from tools.realtime_service import switch_engine
    ap.add_argument("--backend", required=True, choices=MODELS)
    args = ap.parse_args()
    web.run_app(create_app(args.backend, switch=switch_engine),
                host="0.0.0.0", port=PORT, handler_cancellation=True)
