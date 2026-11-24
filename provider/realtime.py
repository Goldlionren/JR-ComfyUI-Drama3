"""Local streaming API for the resident quantized engine. Built with Fish Audio."""

import argparse
import asyncio
import base64
from contextlib import aclosing
import io
import json
import logging
import secrets
import wave

import numpy as np
import regex
from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, web

from jr_realtime_client import IncompleteSpeechError, PORT, PRESET_VOICES, RATE, pcm_event
from jr_realtime_paths import MODELS, PUBLIC_MODELS, engine_binary, model_file
from jr_script_core import MARKS, SPEAKER, apply_to_spoken, chunk_script
from .audio import encode_audio, local_ffmpeg
from .q4_voice import load_q4_voice
from .reference_echo import ReferenceEcho
from .realtime_voices import decode_previous, previous_options, register_voice, voice_options
from .server import LAN_INTERFACE, LAN_NETWORKS, api_boundary, default_db
from .speech import ApiError, FORMATS, catalog

ENGINE_PORT = 8196
MODEL = "drama-3"
SDK_PCM_RATE = 24000
# Match whole emoji clusters (including flags, skin tones and joined family emoji).
EMOJI = regex.compile(r"[#*0-9]\ufe0f?\u20e3|(?=[\p{Extended_Pictographic}\p{Regional_Indicator}\p{Emoji_Modifier}])\X")
SPOKEN_CHAR = regex.compile(r"(?!\p{Default_Ignorable_Code_Point})[\p{L}\p{N}]")
LIST_PREFIX = regex.compile(r"(?m)^(\h*)[0-9]{1,3}[.)]\h+(?=\S)")
EMPHASIS = regex.compile(r"(?<![\w*])(?P<mark>\*{1,3})(?=\S)([^*\n]+?)(?<=\S)(?P=mark)(?![\w*])")


def spoken_text(line):
    return MARKS.sub("", line.replace("<|phoneme_start|>", "").replace("<|phoneme_end|>", ""))


def directed_chunks(text, instructions):
    if not isinstance(instructions, str):
        raise ValueError("instructions must be a short plain-text acting direction.")
    direction = " ".join(instructions.split())
    if len(direction.encode()) > 200 or any(c in direction for c in "[]<>"):
        raise ValueError("Use up to 200 UTF-8 bytes of plain text in instructions. Put inline tags in input.")
    cue = f"[{direction}] " if direction else ""
    # Empty cue and phoneme pairs have no token payload for the script parser.
    text = text.replace("[]", " ").replace("<|phoneme_start|><|phoneme_end|>", " ")
    # Strip list numbering before a period can split off into a separate request.
    text = LIST_PREFIX.sub(r"\1", text)
    text = apply_to_spoken(text, lambda part: EMPHASIS.sub(r"\2", part))
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
        if SPOKEN_CHAR.search(spoken_text(line)):
            output.append((chunk[:offset] + " " if speaker else "") + cue + line)
    return output


async def silent_pcm(rate):
    # A decorations-only request still needs valid audio for WAV previews and SSE clients.
    yield bytes(2 * (rate // 50))


def native_sampling(values, backend):
    if values is None:
        return {}
    limits = {"temperature": (0.05, 2), "top_p": (0.05, 1), "top_k": (0, 4097), "max_frames": (16, 8192)}
    if not isinstance(values, dict) or values.keys() - limits.keys():
        raise ValueError("Unsupported sampling controls.")
    options = {}
    for name, value in values.items():
        low, high = limits[name]
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")
        if name in ("top_k", "max_frames") and type(value) is not int:
            raise ValueError(f"{name} must be an integer.")
        if name == "max_frames":
            if backend == "q4":
                # One Drama 3 codec frame is 2048 samples. Match speech.cpp's floor conversion.
                options["max_seconds"] = value * 2048 / RATE
            else:
                options["max_tokens"] = value
        else:
            options[name] = 4097 if name == "top_k" and value == 0 else value
    return options


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


def create_app(backend, engine_port=ENGINE_PORT, *, voice_db=None, lan=None, extra_networks=(), switch=None, release=None):
    # Two bounded inline WAVs (speaker reference and previous take), encoded as base64.
    app = web.Application(middlewares=[api_boundary], client_max_size=14 * 1024 * 1024)
    if lan is not None:
        app[LAN_INTERFACE] = lan
        app[LAN_NETWORKS] = (lan.network, *extra_networks)
    base = f"http://127.0.0.1:{engine_port}"
    db = voice_db if voice_db is not None else default_db()
    engine = None
    busy = False
    loaded = True

    async def connection(app):
        nonlocal engine
        async with ClientSession(
            timeout=ClientTimeout(total=180), read_bufsize=4 * 1024 * 1024, trust_env=False
        ) as engine:
            yield

    app.cleanup_ctx.append(connection)

    def weight_digest(selected):
        digest = MODELS[selected][1]
        if digest is None:
            try:
                digest = model_file(selected).with_suffix(".gguf.sha256").read_text("ascii").strip()
            except (OSError, UnicodeError):
                return None
        return digest

    # An updated receipt on disk must not relabel audio from the resident weight.
    active_digest = weight_digest(backend)

    def catalog_entry(id_, selected):
        active = loaded and selected == backend
        digest = active_digest if active else weight_digest(selected)
        return {"id": id_, "object": "model", "owned_by": "jr-drama3", "backend": selected, "sha256": digest,
                "active": active,
                "installed": active or (bool(digest) and engine_binary(selected).is_file() and model_file(selected).is_file())}

    async def models(request):
        entries = [catalog_entry(MODEL, backend)]
        entries.extend(catalog_entry(name, key) for name, key in PUBLIC_MODELS.items() if switch or key == backend)
        return web.json_response({"object": "list", "data": entries})

    async def health(request):
        nonlocal loaded
        try:
            async with engine.get(base + "/v1/models", allow_redirects=False) as r:
                if r.status != 200:
                    raise ApiError(503, "Local quantized engine unavailable.")
                items = (await r.json())["data"]
                if not any(m["id"] == MODEL for m in items):
                    raise ApiError(503, "Expected Drama 3 engine on the configured port.")
        except ClientConnectionError as exc:
            raise ApiError(503, "Quantized engine is offline. Submit speech to the managed gateway to load it.") from exc
        loaded = True
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
            nonlocal backend, loaded, active_digest
            digest = active_digest if target == backend and loaded else weight_digest(target)
            try:
                await asyncio.to_thread(switch, target)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ApiError(503, f"Could not load {target}: {exc}") from exc
            backend = target
            active_digest = digest
            loaded = True

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

    async def unload_model(request):
        nonlocal busy, loaded
        if release is None:
            raise ApiError(503, "Start the managed gateway to unload a GGUF.")
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise ApiError(400, "Invalid unload request.")
        if (not isinstance(body, dict) or set(body) != {"model"} or not isinstance(body["model"], str)
                or body["model"] not in PUBLIC_MODELS):
            raise ApiError(400, "Supply the explicit model ID to unload.")
        if busy:
            raise ApiError(429, "Speech engine busy.")
        if PUBLIC_MODELS[body["model"]] != backend:
            raise ApiError(409, "The requested model is not currently selected.")
        async def perform():
            nonlocal loaded, busy
            try:
                await asyncio.to_thread(release, backend)
                loaded = False
            finally:
                busy = False
        busy = True
        task = asyncio.create_task(perform())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise ApiError(503, f"Could not unload model: {exc}")
        return web.json_response(catalog_entry(body["model"], backend))

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

    async def native_sentence(text, voice, part_seed, sampling):
        data = {
            "model": MODEL,
            "input": text,
            **voice,
            "response_format": "pcm",
            "stream_format": "sse",
        }
        if backend != "q4":
            # Keep each validated sentence intact. The engine's default
            # 128-codepoint split can otherwise create an unchecked tail.
            data["options"] = {**sampling, "seed": part_seed, "text_chunk_size": max(300, len(text))}
        else:
            data.update(sampling)
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
                    # audio.cpp's speech SSE omits the task's reached_limit flag.
                    # At the cap the transport alone cannot certify a complete production take.
                    if "max_tokens" in sampling and count >= sampling["max_tokens"] * 2048:
                        raise IncompleteSpeechError("音频达到 max_frames，无法确认完整结束。请缩短台词或提高上限。")
                    break
                count += len(pcm) // 2
                yield pcm
            else:
                raise RuntimeError("Incomplete sentence stream.")

    async def chunks(texts, voice, seed, sampling):
        guard = await asyncio.to_thread(ReferenceEcho.from_voice, voice) if "voice_ref" in voice else None
        if backend == "q4" and "voice_ref" in voice:
            voice = {"voice": await load_q4_voice(engine, base, voice["voice_ref"], voice["reference_text"])}
        for i, text in enumerate(texts):
            part_seed = (seed + i) % 2**32
            if guard is None or not guard.applies(spoken_text(text)):
                async with aclosing(native_sentence(text, voice, part_seed, sampling)) as source:
                    async for pcm in source:
                        yield pcm
                continue

            # Hold this sentence until it passes. Retrying after emitting PCM
            # would leave the reference audible or duplicate earlier dialogue.
            for attempt in range(3):
                attempt_seed = (part_seed + attempt * 0x9E3779B9) % 2**32
                async with aclosing(native_sentence(text, voice, attempt_seed, sampling)) as source:
                    frames = [pcm async for pcm in source]
                if not await asyncio.to_thread(guard.matches, b"".join(frames), RATE):
                    for pcm in frames:
                        yield pcm
                    break
                logging.getLogger(__name__).warning("Reference audio match: %s sentence %d, seed %d, attempt %d/3",
                                                    backend, i + 1, attempt_seed, attempt + 1)
            else:
                raise IncompleteSpeechError(
                    f"第 {i + 1} 句连续三次与参考录音的声学片段匹配。请改写这句或更换参考录音后重试。"
                )

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
                        "instructions", "voice_ref", "reference_text", "sampling", "previous_take"}:
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
            sampling = native_sampling(body.get("sampling"), target)
            previous = decode_previous(body.get("previous_take"), target)
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
            async def generate():
                with previous_options(previous) as history:
                    async with aclosing(chunks(texts, options, seed, {**sampling, **history})) as sequence:
                        async for pcm in sequence:
                            yield pcm
            source = generate()
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
        except IncompleteSpeechError as exc:
            if result is not None and stream == "sse":
                error = {"type": "error", "code": "incomplete_speech", "message": str(exc)}
                await result.write(b"data: " + json.dumps(error).encode() + b"\n\n")
                await result.write_eof()
                return result
            if result is not None and result.prepared and request.transport:
                request.transport.close()
            raise ApiError(502, str(exc), code="incomplete_speech") from exc
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
    app.router.add_post("/v1/models/unload", unload_model)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/audio/voices", list_voices)
    app.router.add_post("/v1/audio/voices", save_voice)
    app.router.add_post("/v1/audio/speech", speech)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    from tools.realtime_service import release_engine, switch_engine
    ap.add_argument("--backend", required=True, choices=MODELS)
    args = ap.parse_args()
    web.run_app(create_app(args.backend, switch=switch_engine, release=release_engine),
                host="0.0.0.0", port=PORT, handler_cancellation=True)
