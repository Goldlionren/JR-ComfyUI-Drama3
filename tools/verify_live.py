"""Exercise true incremental text and PCM delivery through the local Comfy canvas transport."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import time
import uuid

import aiohttp
from verify_comfy import node


async def run(port, destination):
    base = f"http://127.0.0.1:{port}"
    client = uuid.uuid4().hex
    graph = {"1":node("JRDrama3Teacher", directory=""),
             "2":node("JRDrama3LiveText", teacher=["1",0], first_line="Good morning.",
                      seed=42, live=True, idle_seconds=120),
             "3":node("PreviewAny", source=["2",2])}
    async with aiohttp.ClientSession() as session, session.ws_connect(base+"/ws?clientId="+client) as ws:
        async with session.post(base+"/prompt",json={"prompt":graph,"client_id":client}) as r:
            r.raise_for_status()
            pid=(await r.json())["prompt_id"]
        print("Queued incremental text",pid,flush=True)
        token=None
        samples=0
        first=None
        tick=None
        finish_sent=False
        while True:
            msg=await ws.receive(timeout=900)
            if msg.type!=aiohttp.WSMsgType.TEXT: continue
            event=msg.json()
            typ, data=event["type"],event["data"]
            if typ=="jr.drama3.text" and data["active"]:
                token=data["session"]
                tick=time.monotonic()
                async with session.post(base+"/jr-drama3/canvas/text",json={
                    "session":token,"action":"append","text":"Take care."}) as r: r.raise_for_status()
            elif typ=="jr.drama3.audio" and data.get("pcm"):
                samples+=len(base64.b64decode(data["pcm"]))//2
                if first is None: first=time.monotonic()-tick
                if not finish_sent:
                    # Only finish after hearing PCM while the text session is still open.
                    async with session.post(base+"/jr-drama3/canvas/text",json={"session":token,"action":"finish"}) as r:
                        r.raise_for_status()
                    finish_sent=True
            elif typ=="execution_error" and data.get("prompt_id")==pid:
                raise RuntimeError(data["exception_message"])
            elif typ=="execution_success" and data.get("prompt_id")==pid: break
        if not (token and samples and finish_sent):
            raise RuntimeError("No audio arrived while the input session was open")
        for _ in range(20):
            async with session.get(base+"/history/"+pid) as r:
                entry=(await r.json()).get(pid)
            if entry: break
            await asyncio.sleep(0.2)
        report=json.loads(entry["outputs"]["3"]["text"][0])
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(dict(prompt_id=pid, first_pcm_seconds=first,
           preview_samples=samples, closed_after_first_pcm=finish_sent, **report),indent=2),encoding="utf-8")
        print("Verified live playback while text input was open",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=8297)
    p.add_argument("--output",type=Path,default=Path("outputs/teacher-validation/live.json"))
    args=p.parse_args()
    asyncio.run(run(args.port,args.output))
