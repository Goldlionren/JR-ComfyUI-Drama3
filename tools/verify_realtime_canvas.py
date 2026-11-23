"""Run the realtime workflow through ComfyUI and verify its live PCM and final preview."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

import aiohttp
import numpy as np
import soundfile as sf


async def verify(port, output):
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads((root / "example_workflows/07-中英实时语音.json").read_text("utf-8"))
    (node,) = workflow["nodes"]
    text, voice, seed, _, live = node["widgets_values"]
    graph = {
        "1": {"class_type": node["type"], "inputs": dict(text=text, voice=voice, seed=seed, live=live)},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 1]}},
    }
    base = f"http://127.0.0.1:{port}"
    client = uuid.uuid4().hex
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.ws_connect(f"{base}/ws?clientId={client}") as ws:
            start = time.perf_counter()
            async with session.post(base + "/prompt", json={"client_id": client, "prompt": graph}) as resp:
                resp.raise_for_status()
                job = (await resp.json())["prompt_id"]
            received, chunks, first, ended = bytearray(), 0, None, False
            while True:
                msg = await ws.receive(timeout=120)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        continue
                    raise RuntimeError("ComfyUI websocket closed")
                event = msg.json()
                data = event["data"]
                if event["type"] == "jr.drama3.audio":
                    if data.get("end"):
                        ended = True
                    else:
                        assert data["rate"] == 44100
                        received.extend(base64.b64decode(data["pcm"], validate=True))
                        chunks += 1
                        if first is None and len(received) >= 1764:
                            first = time.perf_counter() - start
                elif data.get("prompt_id") == job:
                    if event["type"] in ("execution_error", "execution_interrupted"):
                        raise RuntimeError(data)
                    if event["type"] == "execution_success":
                        break
        async with session.get(base + "/history/" + job) as resp:
            outputs = (await resp.json())[job]["outputs"]
        report = json.loads(outputs["2"]["text"][0])
        async with session.get(base + "/view", params=outputs["1"]["audio"][0]) as resp:
            resp.raise_for_status()
            final, rate = sf.read(io.BytesIO(await resp.read()), dtype="float32")
    preview = np.frombuffer(received, dtype="<i2").astype("float32") / 32768
    assert rate == 44100 and first is not None and ended and chunks > 1 and final.shape == preview.shape
    assert float(np.max(np.abs(preview - final))) <= 2 / 32768
    result = {
        "passed": True,
        "job": job,
        "case": text,
        "queue_to_20ms_pcm_s": first,
        "events": chunks,
        "sample_rate": rate,
        "samples": len(final),
        "live_pcm_sha256": hashlib.sha256(received).hexdigest(),
        "live_matches_saved_audio": True,
        "node_report": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--output", type=Path, default=Path("outputs/realtime-validation/canvas.json"))
    args = parser.parse_args()
    asyncio.run(verify(args.port, args.output))
