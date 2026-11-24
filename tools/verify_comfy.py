"""Run actual local ComfyUI graphs and retain their returned audio and metadata."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def prompt_for(case, model="FP8"):
    text = {"zh": "你好，欢迎回来。", "en": "Welcome home.", "mixed": "你好，Alex。Welcome home.",
            "design": "This is my voice.", "revise": "Welcome home.",
            "clone": "<|speaker:0|> The train is here."}
    graph = {"1": node("JRDrama3Teacher", directory="", model=model),
             "2": node("JRDrama3Sampling", temperature=0.7, top_p=0.9, top_k=0, max_frames=180)}
    def generate(text, mode="line", direction="", seed=184, **other):
        return node("JRDrama3Generate", teacher=["1",0], settings=["2",0], text=text, seed=seed,
                    live=False, mode=mode, direction=direction, **other)
    if case in ("zh","en","mixed"):
        graph["3"] = generate(text[case])
    elif case == "scene":
        graph["6"] = node("JRDrama3Script", script="Lin: Hi!\nBo: Hey!", scene="Two friends meet.",
                          biographies='{"Lin":"a young woman","Bo":"a cheerful man"}')
        graph["3"] = generate(["6",2], "scene", cast=["6",1])
    elif case == "design":
        graph["3"] = generate(text[case], "design", "A warm adult narrator with a gentle voice.")
    elif case == "clone":
        graph["6"] = generate(text["en"])
        graph["7"] = node("JRDrama3Reference", name="A", speaker=0, transcript=text["en"],
                          description="", language="en", audio=["6",0])
        graph["3"] = generate(text[case], cast=["7",0])
    elif case == "revise":
        graph["6"] = generate(text["en"])
        graph["3"] = generate(text["revise"], "revise", "Softer and more slowly than before.",
                              previous=["6",1], seed=185)
    elif case == "infill":
        graph["6"] = generate(text["en"])
        graph["3"] = node("JRDrama3Infill", teacher=["1",0], source=["6",0],
                          revised_transcript="Welcome back.", start_seconds=0.32, end_seconds=1.06,
                          seed=184, settings=["2",0])
    else:
        raise ValueError(case)
    graph["4"] = node("PreviewAudio", audio=["3",0])
    graph["5"] = node("PreviewAny", source=["3",2])
    return graph


def execute(base, graph, dest, case):
    body = json.dumps({"prompt":graph}).encode()
    try:
        with urlopen(Request(base+"/prompt", data=body, headers={"Content-Type":"application/json"}),timeout=20) as r:
            pid = json.load(r)["prompt_id"]
    except HTTPError as err:
        raise RuntimeError(err.read().decode()) from err
    print(f"Queued {case} {pid}",flush=True)
    start=time.monotonic()
    while time.monotonic()-start < 1800:
        with urlopen(base+"/history/"+pid, timeout=15) as r:
            entry=json.load(r).get(pid)
        if entry: break
        time.sleep(2)
    else: raise TimeoutError(pid)
    if entry["status"]["status_str"] != "success":
        error=next((v for k,v in entry["status"]["messages"] if k=="execution_error"),entry["status"])
        raise RuntimeError(str(error.get("exception_message",error)))
    outputs=entry["outputs"]
    record=json.loads(outputs["5"]["text"][0])
    descriptor=outputs["4"]["audio"][0]
    with urlopen(base+"/view?"+urlencode(descriptor), timeout=60) as r:
        content=r.read()
    model = graph["1"]["inputs"].get("model", "FP8")
    expected = "eos" if model == "FP8" else "complete"
    if record.get("finish_reason") != expected or (model != "FP8" and record.get("model") != model):
        raise RuntimeError(f"{case}: unexpected model or finish status for {model} (prompt {pid}): {record}")
    if record.get("seconds",0) <= 0 or not content:
        raise RuntimeError(f"{case}: empty output (prompt {pid})")
    dest.mkdir(parents=True,exist_ok=True)
    (dest / (case+Path(descriptor["filename"]).suffix)).write_bytes(content)
    record["prompt_id"]=pid
    (dest / (case+".json")).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(case, json.dumps({k:record.get(k) for k in ("finish_reason","seconds","elapsed_seconds","real_time_factor",
                                                         "host_expert_layers")}), flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases",nargs="+",choices=["zh","en","mixed","scene","clone","design","revise","infill"])
    parser.add_argument("--port",type=int,default=8297)
    parser.add_argument("--model", choices=["FP8", "drama-3-5lang", "drama-3-nvfp4", "drama-3-q4"], default="FP8")
    parser.add_argument("--dest",type=Path,default=Path("outputs/teacher-validation"))
    args=parser.parse_args()
    for case in args.cases:
        execute(f"http://127.0.0.1:{args.port}", prompt_for(case, args.model), args.dest, case)
