"""Encode completed speech in the audio format requested by the client."""
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .speech import ApiError


async def convert(flac, fmt, speed, executable):
    if not flac.startswith(b"fLaC"):
        raise ApiError(502, "ComfyUI returned invalid FLAC data.")
    return await encode_audio(flac, "flac", fmt, speed, executable)


def local_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise ApiError(503, "Audio encoder unavailable. Install imageio-ffmpeg in the gateway Python environment.") from exc


async def encode_audio(audio, input_format, fmt, speed, executable):
    args = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-protocol_whitelist", "pipe",
            "-f", input_format, "-i", "pipe:0", "-map", "0:a:0", "-map_metadata", "-1", "-ac", "1"]
    if speed != 1.:
        filters = []
        while speed > 2:
            filters.append("atempo=2")
            speed /= 2
        while speed < .5:
            filters.append("atempo=0.5")
            speed *= 2
        filters.append(f"atempo={speed:.9g}")
        args.extend(["-af", ",".join(filters)])
    encoders = {"mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
                "opus": ["-ar", "48000", "-c:a", "libopus", "-b:a", "64k", "-f", "opus"],
                "aac": ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
                "flac": ["-c:a", "flac", "-f", "flac"],
                "wav": ["-ar", "44100", "-c:a", "pcm_s16le", "-f", "wav"],
                "pcm": ["-ar", "24000", "-c:a", "pcm_s16le", "-f", "s16le"]}
    # Seekable output lets FFmpeg finalize WAV sizes, FLAC STREAMINFO and
    # MP3 duration headers before returning to arbitrary local audio players.
    with TemporaryDirectory(prefix="jr-drama3-") as work:
        output = Path(work) / ("speech." + fmt)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, *encoders[fmt], str(output), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            raise ApiError(503, f"Could not start local FFmpeg: {exc}.") from exc
        try:
            _, error = await proc.communicate(audio)
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.communicate()
        if proc.returncode or not output.is_file() or not output.stat().st_size:
            detail = error.decode(errors="replace").strip()[:400]
            raise ApiError(502, f"FFmpeg failed to encode audio: {detail}")
        return await asyncio.to_thread(output.read_bytes)
