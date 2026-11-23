"""Submit exactly one prompt to an existing loopback ComfyUI instance."""
import asyncio
import ipaddress
import logging
import uuid
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientTimeout

from .speech import ApiError, build_graph, completed_audio


def loopback_url(value):
    parsed = urlsplit(value)
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    if (parsed.scheme != "http" or not host or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("ComfyUI URL must be a local HTTP origin such as http://127.0.0.1:8188.")
    ip = ipaddress.IPv4Address(host)
    if not ip.is_loopback:
        raise ValueError("ComfyUI must listen on IPv4 loopback.")
    port = parsed.port
    if port == 0:
        raise ValueError("Invalid ComfyUI port.")
    return f"http://{ip}" + (f":{port}" if port else "")


async def ready(session, base):
    try:
        async with session.get(base + "/object_info/JRDrama3Generate", allow_redirects=False,
                               timeout=ClientTimeout(total=3)) as response:
            return response.status == 200 and "JRDrama3Generate" in await response.json()
    except (ClientError, TimeoutError, ValueError, TypeError):
        return False


async def select(session, primary, fallback):
    if await ready(session, primary):
        return primary
    if fallback and await ready(session, fallback):
        return fallback
    raise ApiError(503, "No configured loopback ComfyUI backend has JRDrama3Generate available.",
                   code="backend_unavailable")


async def fetch_json(session, method, url, **kwargs):
    async with session.request(method, url, allow_redirects=False, **kwargs) as response:
        if not 200 <= response.status < 300:
            detail = (await response.content.read(1000)).decode(errors="replace")
            raise ApiError(502, f"ComfyUI returned HTTP {response.status}: {detail[:400]}")
        try:
            result = await response.json()
        except (ValueError, ClientError) as exc:
            raise ApiError(502, "ComfyUI returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise ApiError(502, "Unexpected response from ComfyUI.")
        return result


async def cancel_submission(session, base, submit, proposed):
    # A client can disconnect during validation before /prompt returns. Wait
    # for its bounded acknowledgement to avoid cancelling before it is queued.
    try:
        response = await submit
        job = str(uuid.UUID(response["prompt_id"]))
    except (ClientError, TimeoutError, ValueError, KeyError, TypeError, ApiError):
        job = proposed
    try:
        # The installed Aki checks the running job ID atomically. Never use a
        # global interrupt, free, clear-queue or clear-history operation.
        await fetch_json(session, "POST", base + f"/api/jobs/{job}/cancel", timeout=ClientTimeout(total=5))
    except (ClientError, TimeoutError, ApiError, ValueError) as exc:
        logging.warning("Could not cancel local ComfyUI job %s: %s", job, exc)


async def synthesize(session, base, params, *, unload=False, poll=.4):
    graph, reports = build_graph(params, unload)
    proposed = str(uuid.uuid4())
    submit = asyncio.create_task(fetch_json(session, "POST", base + "/prompt", json={
        "prompt_id": proposed, "client_id": str(uuid.uuid4()), "prompt": graph,
    }, timeout=ClientTimeout(total=20)))
    finished = False
    try:
        response = await asyncio.shield(submit)
        try:
            job = str(uuid.UUID(response["prompt_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(502, "ComfyUI did not acknowledge the prompt ID.") from exc
        while True:
            history = await fetch_json(session, "GET", base + "/history/" + job)
            if job in history:
                finished = True
                break
            await asyncio.sleep(poll)
        descriptor = completed_audio(history[job], reports, params["voice"])
        async with session.get(base + "/view", params=descriptor, allow_redirects=False) as audio:
            if audio.status != 200:
                raise ApiError(502, "Cannot read the completed ComfyUI preview.")
            return await audio.read(), job
    finally:
        if not finished:
            await asyncio.shield(cancel_submission(session, base, submit, proposed))
