"""Pinned quantized models and their local inference engines."""
import os
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / ".runtime"
MODELS = {
    "q4": ("Drama-35B-A3B-v3-Q4_K.gguf", "2a57b14c9b208a2ff833ab7305b920ca8f154ad807004a52d96a86a90a55523e"),
    "nvfp4": (
        "Drama3-V2-78e5ab7-NVFP4-256E.gguf",
        "5577beb99be7bc0f4c8324daea90e2f3c8765d17842db41016560fb4820312b1",
    ),
}


def engine_binary(backend):
    if backend == "q4":
        return RUNTIME / "speechcpp" / ("speech.exe" if os.name == "nt" else "speech")
    build = "windows-cuda-release" if os.name == "nt" else "linux-cuda-release"
    binary = "audiocpp_server.exe" if os.name == "nt" else "audiocpp_server"
    return RUNTIME / "audio.cpp" / "build" / build / "bin" / binary


def model_file(backend):
    key = f"JR_DRAMA3_{backend.upper()}_MODEL"
    override = os.environ.get(key)
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError(f"{key} must be an absolute path to the GGUF file.")
        return path
    return RUNTIME / MODELS[backend][0]
