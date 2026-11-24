"""Read complete, mono 44.1 kHz PCM streams from the local Drama 3 engines."""

import base64
from http.client import HTTPConnection
import json
import re

RATE = 44100
PORT = 8195
MAX_REF_BYTES = 5 * 1024 * 1024
PRESET_VOICES = ("zh", "en", "ja", "ko", "yue")


def is_saved_voice(voice):
    return isinstance(voice, str) and re.fullmatch(r"[0-9a-f]{64}", voice) is not None


def pcm_event(event, sample_count):
    kind = event.get("type")
    if kind == "speech.audio.delta":
        pcm = base64.b64decode(event["audio"], validate=True)
        if not pcm or len(pcm) % 2:
            raise RuntimeError("Invalid PCM sample count.")
        return pcm
    if kind == "speech.audio.done":
        if (
            not sample_count
            or event.get("stop", "complete") != "complete"
            or event.get("samples", sample_count) != sample_count
        ):
            raise RuntimeError(f"Incomplete speech: {event}")
        return None
    raise RuntimeError(f"Unexpected speech event: {str(event)[:500]}")


def stream_speech(text, voice, seed, *, port=PORT, instructions="", reference_wav=None, reference_text="",
                  model="drama-3"):
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError("台词必须为字符串，最多 4096 个字符。")
    if voice not in PRESET_VOICES and not is_saved_voice(voice):
        raise ValueError("请选择内置音色或已保存的完整音色 ID。")
    conn = HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        conn.request("GET", "/v1/models")
        models = conn.getresponse()
        if models.status != 200:
            raise RuntimeError(f"量化服务返回 HTTP {models.status}")
        entry = next((entry for entry in json.load(models)["data"] if entry["id"] == model), None)
        if entry is None:
            raise ValueError(f"服务没有提供模型 {model}，请检查模型安装和服务版本。")
        public = entry.get("owned_by") == "jr-drama3"
        if not public and not text.strip():
            raise ValueError("请通过实时 API 跳过空台词。")
        if not public and (instructions or reference_wav is not None or reference_text or is_saved_voice(voice)):
            raise ValueError(f"自定义音色和整段语气请通过本地实时服务端口 {PORT} 使用。")
        req = {
            "model": model,
            "input": text,
            "voice": voice,
            "stream_format": "sse",
            "response_format": "pcm",
        }
        if instructions:
            req["instructions"] = instructions
        if reference_wav is not None:
            if len(reference_wav) > MAX_REF_BYTES:
                raise ValueError("参考 WAV 最大为 5 MiB，请先裁剪参考录音。")
            req["voice_ref"] = {"type": "base64", "data": base64.b64encode(reference_wav).decode("ascii")}
        if reference_text:
            req["reference_text"] = reference_text
        if "speech" in entry or public:
            req["seed"] = seed
        else:
            req["options"] = {"seed": seed}
        conn.request(
            "POST", "/v1/audio/speech", json.dumps(req).encode(), {"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(2048).decode(errors='replace')}")
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise RuntimeError("量化服务没有返回流式音频事件。")
        sample_count = 0
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            pcm = pcm_event(json.loads(data), sample_count)
            if pcm is None:
                return
            sample_count += len(pcm) // 2
            yield pcm
        raise RuntimeError("音频流在完成事件前中断。")
    except ConnectionRefusedError as err:
        raise RuntimeError("请先运行 tools/realtime_service.py start 启动本地量化服务。") from err
    finally:
        conn.close()
