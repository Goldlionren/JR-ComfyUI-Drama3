"""Verify the deployed localhost API using real teacher audio and offline recognition."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def run(port, dest):
    base = f"http://127.0.0.1:{port}"
    def get(path):
        with urlopen(base + path, timeout=15) as response:
            return json.load(response)
    def speak(body):
        request = Request(base + "/v1/audio/speech", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        return urlopen(request, timeout=900)

    dest.mkdir(parents=True, exist_ok=True)
    info = {"health": get("/health"), "models": get("/v1/models"), "runs": []}
    assert any(x["id"] == "drama-3" for x in info["models"]["data"]), info["models"]
    get("/v1/audio/voices")
    try:
        speak({"model":"drama-3", "input":"Hello.", "voice":"does-not-exist-123", "response_format":"wav"})
    except HTTPError as err:
        error = json.loads(err.read())
        assert 400 <= err.code < 500 and "error" in error, (err.code, error)
        info["unknown_voice_http"] = err.code
    else:
        raise AssertionError("Unknown voice was silently accepted.")

    pkg = types.ModuleType("jr_provider_check")
    pkg.__path__ = [str(ROOT)]
    sys.modules[pkg.__name__] = pkg
    from jr_provider_check.jr_analysis_asr import transcribe
    from jr_provider_check.jr_analysis_quality import review

    for language, text in [("en", "Welcome home."), ("zh", "你好，欢迎回来。")]:
        start = time.monotonic()
        with speak({"model": "drama-3", "input": text, "voice": "alloy", "response_format": "wav",
                    "seed": 184}) as response:
            payload = response.read()
            media = response.headers.get_content_type()
        elapsed = time.monotonic() - start
        path = dest / f"provider-{language}.wav"
        path.write_bytes(payload)
        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 44100 and wave.shape[1] == 1 and wave.shape[0] > rate//8
        audio = dict(waveform=torch.from_numpy(wave.T.copy())[None], sample_rate=rate)
        heard = transcribe(audio, "Whisper small", language=language, device="cpu")["text"]
        quality = review(audio, text, heard)
        entry = dict(language=language, target=text, recognized=heard, seconds=len(wave) / rate,
                     elapsed_seconds=elapsed, sample_rate=rate, content_type=media, cer=quality["cer"],
                     word_or_mixed_error=quality.get("wer", quality.get("mixed_token_error_rate")),
                     sha256=hashlib.sha256(payload).hexdigest())
        info["runs"].append(entry)
        (dest / "provider-validation.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
        print(language, heard, entry["cer"], flush=True)
        assert quality["cer"] == 0, entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--dest", type=Path, default=ROOT / "outputs/provider-validation")
    args = parser.parse_args()
    run(args.port, args.dest)
