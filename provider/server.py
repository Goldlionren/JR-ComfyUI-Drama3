"""Local HTTP endpoints. The process owns no model or accelerator state."""
import asyncio
import ipaddress
import logging
from pathlib import Path
import sqlite3
import sys
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from jr_voice_store import library_path

from .audio import convert
from .backend import loopback_url, ready, select, synthesize
from .speech import ApiError, FORMATS, MODEL, VOICES, catalog, speech_params

LAN_INTERFACE = web.AppKey("lan_interface", ipaddress.IPv4Interface)
LAN_NETWORKS = web.AppKey("lan_networks", tuple)


def loopback_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.IPv4Address(host).is_loopback
    except (ValueError, TypeError):
        return False


def error_response(status, message, param=None, code=None):
    kind = ("rate_limit_error" if status == 429 else "server_error" if status >= 500 else "invalid_request_error")
    return web.json_response({"error": {"message": message, "type": kind, "param": param, "code": code}},
                             status=status, headers={"Retry-After": "5"} if status == 429 else None)


@web.middleware
async def api_boundary(request, handler):
    try:
        lan = request.app.get(LAN_INTERFACE)
        host = request.url.host
        local = loopback_host(request.remote) and loopback_host(host)
        if lan is not None and not local:
            if host != str(lan.ip) or not request.remote:
                raise ApiError(403, "Request outside the configured network.")
            peer = ipaddress.ip_address(request.remote)
            if not any(peer in network for network in request.app[LAN_NETWORKS]):
                raise ApiError(403, "Request outside the configured network.")
        origin = request.headers.get("Origin")
        if origin:
            origin_host = urlsplit(origin).hostname
            if not loopback_host(origin_host) and origin_host != (str(lan.ip) if lan else host):
                raise ApiError(403, "Origin outside the configured network.")
        return await handler(request)
    except ApiError as exc:
        return error_response(exc.status, str(exc), exc.param, exc.code)
    except web.HTTPException as exc:
        return error_response(exc.status, exc.reason)
    except TimeoutError:
        return error_response(504, "The local speech request timed out.", code="timeout")
    except (ClientError, OSError, sqlite3.Error) as exc:
        return error_response(503, f"Local speech service unavailable: {exc}", code="backend_unavailable")
    except Exception:
        logging.exception("Local speech request failed")
        return error_response(500, "The local speech request failed.", code="internal_error")


def model_info():
    return {"object": "model", "id": MODEL, "created": 0, "owned_by": "local"}


def default_db():
    root = Path(__file__).resolve().parents[1]
    aki = Path(sys.executable).resolve().parent.parent / "ComfyUI"
    for comfy in (aki, root.parent.parent if root.parent.name == "custom_nodes" else None):
        if comfy and (comfy / "main.py").is_file():
            return library_path(comfy / "user/jr_drama3/voices.sqlite3")
    return library_path(root / ".runtime/provider-service/voices.sqlite3")


def create_app(*, primary="http://127.0.0.1:8188", fallback=None, voice_db=None, ffmpeg="ffmpeg",
               timeout=1800, max_concurrent=1, poll=.4):
    primary = loopback_url(primary)
    fallback = loopback_url(fallback) if fallback else None
    if fallback == primary:
        raise ValueError("Primary and fallback must have different ports or addresses.")
    if not 0 < timeout <= 86400 or not 1 <= max_concurrent <= 16:
        raise ValueError("timeout must be in (0,86400], max_concurrent must be in 1–16.")
    voice_db = Path(voice_db) if voice_db else default_db()
    app = web.Application(middlewares=[api_boundary], client_max_size=64*1024)
    session = None
    active = 0
    last_backend = None

    async def connections(app):
        nonlocal session
        async with ClientSession(timeout=ClientTimeout(total=30), trust_env=False) as session:
            yield
    app.cleanup_ctx.append(connections)

    async def health(request):
        endpoints = [url for url in (primary, fallback) if url]
        checks = dict(zip(endpoints, await asyncio.gather(*(ready(session, url) for url in endpoints))))
        selected = next((url for url in endpoints if checks[url]), None)
        return web.json_response({"status": "ok" if selected else "unavailable", "model": MODEL,
                                 "primary": primary, "fallback": fallback, "backends": checks,
                                 "selected_backend": selected, "last_backend": last_backend,
                                 "active": active, "max_concurrent": max_concurrent}, status=200 if selected else 503)

    async def models(request):
        return web.json_response({"object": "list", "data": [model_info()]})

    async def voices(request):
        builtin = [{"id": key, "object": "voice", "source": "local_design", "description": desc}
                   for key, desc in VOICES.items()]
        return web.json_response({"object": "list", "data": builtin + await asyncio.to_thread(catalog, voice_db)})

    async def speech(request):
        nonlocal active, last_backend
        if request.content_type != "application/json":
            raise ApiError(415, "Use Content-Type: application/json.")
        try:
            params = speech_params(await request.json())
        except (ValueError, UnicodeError) as exc:
            raise ApiError(400, "Invalid JSON request body.") from exc
        if active >= max_concurrent:
            raise ApiError(429, "Local speech capacity is in use. Retry later.", code="rate_limit_exceeded")
        active += 1
        try:
            async with asyncio.timeout(timeout):
                voice = params["voice"]
                if voice not in VOICES and not await asyncio.to_thread(catalog, voice_db, voice):
                    raise ApiError(400, "Saved voice ID was not found in the local voice library.", param="voice")
                base = await select(session, primary, fallback)
                last_backend = base
                source, job = await synthesize(session, base, params, unload=base == fallback, poll=poll)
                output = await convert(source, params["fmt"], params["speed"], ffmpeg)
                return web.Response(body=output, content_type=FORMATS[params["fmt"]], headers={
                    "X-Request-ID": job, "X-Drama3-Backend": base, "X-Drama3-Seed": str(params["seed"]),
                    "Cache-Control": "no-store"})
        finally:
            active -= 1

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/v1/audio/voices", voices)
    app.router.add_post("/v1/audio/speech", speech)
    return app
