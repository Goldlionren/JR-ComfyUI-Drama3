import argparse

from aiohttp import web
from .server import create_app, default_db


def main():
    parser = argparse.ArgumentParser(description="Local Drama 3 OpenAI-compatible speech API")
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--fallback-comfy-url", help="Standby loopback ComfyUI with the same model and voice library.")
    parser.add_argument("--voice-db", default=str(default_db()), help="ComfyUI user/jr_drama3/voices.sqlite3")
    parser.add_argument("--timeout", type=float, default=1800, help="Maximum seconds per speech request.")
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Local FFmpeg executable.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1–65535")
    try:
        app = create_app(primary=args.comfy_url, fallback=args.fallback_comfy_url, voice_db=args.voice_db,
                         timeout=args.timeout, max_concurrent=args.max_concurrent, ffmpeg=args.ffmpeg)
    except ValueError as exc:
        parser.error(str(exc))
    web.run_app(app, host="127.0.0.1", port=args.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
