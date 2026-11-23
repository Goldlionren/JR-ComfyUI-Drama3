"""Load a transcribed recording into the running speech.cpp Drama 3 model."""
import asyncio
import base64
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from aiohttp import FormData

from jr_realtime_paths import engine_binary, model_file
from .speech import ApiError, MODEL


async def encode_reference(wav, transcript):
    with TemporaryDirectory(prefix="jr-drama3-") as directory:
        source = Path(directory) / "reference.wav"
        output = Path(directory) / "reference.voice.gguf"
        source.write_bytes(wav)
        process = await asyncio.create_subprocess_exec(
            engine_binary("q4"), "voice", model_file("q4"), source, output,
            "--transcript", transcript, "--device", "cpu",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, errors = await process.communicate()
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        if process.returncode:
            message = errors[-1024:].decode("utf-8", errors="replace").strip()
            raise ApiError(502, f"Q4 reference encoding failed: {message}")
        return output.read_bytes()


async def engine_has_voice(engine, base, name):
    async with engine.get(base + "/v1/models", allow_redirects=False) as response:
        response.raise_for_status()
        models = (await response.json())["data"]
    model = next((entry for entry in models if entry["id"] == MODEL), None)
    if model is None:
        raise ApiError(503, "Expected the Drama 3 Q4 engine on the configured port.")
    return any(voice["name"] == name for voice in model["speech"]["voices"])


async def load_q4_voice(engine, base, ref, transcript):
    wav = base64.b64decode(ref["data"], validate=True)
    digest = sha256(sha256(wav).digest() + transcript.encode("utf-8")).hexdigest()
    name = f"jr-{digest}"
    # Native registrations follow the lifetime of the model, independently of the gateway.
    if await engine_has_voice(engine, base, name):
        return name

    encoded = await encode_reference(wav, transcript)
    form = FormData()
    form.add_field("name", name)
    form.add_field("file", encoded, filename="reference.voice.gguf", content_type="application/octet-stream")
    async with engine.post(base + "/speech/voices", data=form, allow_redirects=False) as response:
        if response.status == 200:
            return name
        status = response.status
        detail = (await response.content.read(1024)).decode(errors="replace")
    # A disconnected request can still complete its upload inside speech.cpp.
    if status == 400 and await engine_has_voice(engine, base, name):
        return name
    raise ApiError(502, f"Q4 engine could not add the reference voice: {detail}")
