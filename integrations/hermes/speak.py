"""Run the installed Hermes TTS tool with an isolated, local Drama 3 profile."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


PROFILE_CONFIG = Path(__file__).with_name("config.yaml")
WORKSPACE_PROFILE = Path(__file__).resolve().parents[2] / ".runtime" / "hermes-drama3"
LOCAL_BASE = "http://127.0.0.1:8193/v1"


def speak(text, output, *, instructions=None):
    # Import after setting HERMES_HOME. Hermes resolves some paths at import time.
    from hermes_cli.config import load_config
    from tools import tts_tool  # registers text_to_speech in Hermes

    config = load_config()["tts"]
    if (config["provider"], config["openai"]["base_url"], config["openai"]["model"]) != (
        "openai", LOCAL_BASE, "drama-3"
    ):
        raise ValueError("Hermes did not load the local Drama 3 profile.")

    if output.suffix.lower() == ".pcm":
        if instructions:
            raise ValueError("Hermes' PCM streamer accepts text only. Use WAV for instructions.")
        from tools.tts_streaming import resolve_streaming_provider

        streamer = resolve_streaming_provider(config)
        if streamer is None:
            raise RuntimeError("The Hermes OpenAI-compatible PCM streamer is unavailable.")
        started = time.monotonic()
        first_chunk = None
        count = 0
        with output.open("wb") as target:
            for chunk in streamer.stream(text):
                if not chunk:
                    continue
                if first_chunk is None:
                    first_chunk = time.monotonic() - started
                count += 1
                target.write(chunk)
        if first_chunk is None or output.stat().st_size % 2:
            raise ValueError("Empty or incomplete PCM response.")
        return {"provider": "openai", "route": "Hermes PCM streamer",
                "sample_rate": streamer.sample_rate, "channels": 1, "encoding": "s16le",
                "received_chunks": count, "first_bytes_seconds": round(first_chunk, 3)}

    arguments = {"text": text, "output_path": str(output)}
    if instructions:
        arguments["instructions"] = instructions
    result = tts_tool.registry.dispatch("text_to_speech", arguments)
    if isinstance(result, str):
        result = json.loads(result)
    if not result.get("success"):
        raise RuntimeError(result.get("error", "Hermes text_to_speech failed."))
    if result.get("provider") != "openai" or Path(result["file_path"]).resolve() != output:
        raise RuntimeError("Hermes delivered an unexpected provider or output path.")
    return {"provider": result["provider"], "route": "Hermes registry text_to_speech",
            "chunks": result.get("chunk_count", 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("output", type=Path, help="New output file (.wav, .mp3, .ogg or .pcm)")
    parser.add_argument("--instructions", help="Optional voice direction for file synthesis")
    parser.add_argument("--workspace-profile", action="store_true",
                        help="Use the dedicated .runtime/hermes-drama3 profile and its selected voice")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix not in {".wav", ".mp3", ".ogg", ".pcm"}:
        parser.error("Use a lowercase .wav, .mp3, .ogg or .pcm suffix.")
    if not args.text.strip():
        parser.error("The input must contain speech.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage output in a private directory so failed inference leaves the destination untouched.
    with tempfile.TemporaryDirectory(prefix="drama3-hermes-") as temporary:
        staging = Path(temporary)
        home = WORKSPACE_PROFILE if args.workspace_profile else staging
        if args.workspace_profile:
            if not (home / "config.yaml").is_file():
                parser.error(f"Dedicated profile is missing: {home}")
        else:
            shutil.copyfile(PROFILE_CONFIG, home / "config.yaml")
        os.environ["HERMES_HOME"] = str(home)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        for variable in ("HERMES_PROFILE", "HERMES_CONFIG", "HERMES_CONFIG_PATH", "HERMES_ENV",
                         "HERMES_ENV_PATH", "HERMES_SESSION_PLATFORM"):
            os.environ.pop(variable, None)
        started = time.monotonic()
        staged = staging / output.name
        if output.exists():
            parser.error(f"Output already exists: {output}")
        result = speak(args.text, staged, instructions=args.instructions)
        if not staged.is_file() or not staged.stat().st_size:
            raise RuntimeError("Hermes returned no audio.")
        with staged.open("rb") as source, output.open("xb") as target:
            shutil.copyfileobj(source, target)
        result.update(hermes=importlib.metadata.version("hermes-agent"),
                      model="drama-3", endpoint=LOCAL_BASE + "/audio/speech",
                      text=args.text, output=str(output), bytes=output.stat().st_size,
                      elapsed_seconds=round(time.monotonic() - started, 3))
        with output.open("rb") as audio:
            result["sha256"] = hashlib.file_digest(audio, "sha256").hexdigest()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
