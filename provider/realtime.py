"""Local streaming API for the resident quantized engine. Built with Fish Audio."""

import argparse
import asyncio
import base64
from contextlib import aclosing
import io
from ipaddress import IPv4Interface, IPv4Network
import json
from pathlib import Path
import secrets
import wave

import numpy as np
import regex
from aiohttp import ClientSession, ClientTimeout, web

from jr_realtime_client import PORT, RATE, pcm_event
from jr_script_core import MARKS, SPEAKER, apply_to_spoken, chunk_script
from .q4_voice import load_q4_voice
from .realtime_voices import register_voice, voice_options
from .server import LAN_INTERFACE, LAN_NETWORKS, api_boundary, default_db
from .speech import ApiError, catalog

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


def create_app(backend, engine_port=ENGINE_PORT, *, voice_db=None, lan=None, extra_networks=()):
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

    async def models(request):
        async with engine.get(base + "/v1/models", allow_redirects=False) as r:
            if r.status != 200:
                raise ApiError(503, "Local quantized engine unavailable.")
            items = (await r.json())["data"]
            if not any(m["id"] == MODEL for m in items):
                raise ApiError(503, "Expected Drama 3 engine on the configured port.")
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": MODEL, "object": "model", "owned_by": "jr-drama3", "backend": backend}],
            }
        )

    async def list_voices(request):
        rows = [{"id": key, "object": "voice", "source": "preset"} for key in ("zh", "en")]
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
            if backend == "nvfp4":
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
                    pcm = pcm_event(json.loads(payload), count)
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
        if not isinstance(body, dict) or body.get("model", MODEL) != MODEL:
            raise ApiError(400, "Expected a drama-3 speech request.")
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
        if fmt not in ("pcm", "wav") or stream not in ("audio", "sse") or (fmt == "wav" and stream == "sse"):
            raise ApiError(400, "Choose pcm (audio or sse) or wav (audio).")
        try:
            instructions = body.get("instructions", "")
            texts = directed_chunks(text, instructions)
            if texts:
                options = await asyncio.to_thread(voice_options, body, backend, db)
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
            if fmt == "wav":
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as w:
                    w.setparams((1, 2, RATE, 0, "NONE", ""))
                    async for pcm in source:
                        w.writeframesraw(pcm)
                return web.Response(body=buffer.getvalue(), content_type="audio/wav", headers=headers)
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
    app.router.add_get("/health", models)
    app.router.add_get("/v1/audio/voices", list_voices)
    app.router.add_post("/v1/audio/voices", save_voice)
    app.router.add_post("/v1/audio/speech", speech)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["q4", "nvfp4"])
    args = ap.parse_args()
    config = Path(__file__).resolve().parents[1] / ".runtime/provider-service/realtime-lan.txt"
    lines = config.read_text("utf-8").strip().splitlines() if config.is_file() else []
    lan = IPv4Interface(lines[0]) if lines else None
    extra_networks = tuple(IPv4Network(line.strip()) for line in lines[1:] if line.strip())
    hosts = ["127.0.0.1", str(lan.ip)] if lan else ["127.0.0.1"]
    web.run_app(create_app(args.backend, lan=lan, extra_networks=extra_networks),
                host=hosts, port=PORT, handler_cancellation=True)
