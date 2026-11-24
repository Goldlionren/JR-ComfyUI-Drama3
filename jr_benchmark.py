"""Repeatable local Chinese, English and code-switching measurements."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io
from .jr_benchmark_cases import CASES, FIXTURES, REFERENCES


SUITE = "jr-drama3-zh-en-v1"
POLL_SECONDS = 0.1


def fixed_reference(key):
    filename, digest, transcript, voice = REFERENCES[key]
    path = FIXTURES / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The fixed benchmark reference has changed: {path.name}")
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2:
            raise ValueError("The fixed reference must be PCM 16-bit")
        sample_rate, channels = audio.getframerate(), audio.getnchannels()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    waveform = torch.from_numpy(samples.reshape(-1, channels).T.astype("float32") / 32768).unsqueeze(0)
    data = {"waveform": waveform, "sample_rate": sample_rate}
    return [{"name": "Reference", "speaker": 0, "transcript": transcript, "audio": data}], {
        "file": filename, "sha256": digest, "transcript": transcript, "voice": voice, "kind": "offline synthetic speech"}


class MemorySampler:
    """Sample system-wide counters without creating CUDA contexts or resetting other nodes' stats."""

    def __init__(self):
        self.stop = threading.Event()
        self.samples = 0
        self.errors = set()
        self.devices = []
        self.gpu = None
        self.ps = None
        self.peaks = {}
        self.thread = None

    def __enter__(self):
        try:
            import psutil
            self.ps = psutil
        except ImportError:
            self.errors.add("psutil unavailable, RAM counters unavailable")
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu = pynvml
            for i in range(pynvml.nvmlDeviceGetCount()):
                device = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.devices.append((device, str(pynvml.nvmlDeviceGetUUID(device)), str(pynvml.nvmlDeviceGetName(device))))
        except Exception as err:
            self.errors.add(f"Device memory counter unavailable ({type(err).__name__})")
        self._sample()
        self.initial = dict(self.peaks)
        self.thread = threading.Thread(target=self._loop, name="jr-drama3-benchmark-memory", daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        def update(key, value):
            self.peaks[key] = max(self.peaks.get(key, 0), int(value))
        if self.ps is not None:
            try:
                vm = self.ps.virtual_memory()
                update("system_used", vm.total - vm.available)
                root = self.ps.Process()
                rss = 0
                for proc in [root, *root.children(recursive=True)]:
                    try:
                        rss += proc.memory_info().rss
                    except (self.ps.NoSuchProcess, self.ps.AccessDenied):
                        self.errors.add("A child process was inaccessible during RSS sampling")
                update("process_tree_rss", rss)
            except Exception as err:
                self.errors.add(f"RAM sampling unavailable ({type(err).__name__})")
        for device, uid, _ in self.devices:
            try:
                update(uid, self.gpu.nvmlDeviceGetMemoryInfo(device).used)
            except Exception as err:
                self.errors.add(f"NVML sample unavailable for {uid} ({type(err).__name__})")
        self.samples += 1

    def _loop(self):
        while not self.stop.wait(POLL_SECONDS):
            self._sample()

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self._sample()
        if self.gpu is not None:
            self.gpu.nvmlShutdown()

    def report(self):
        return {"units": "bytes", "interval_seconds": POLL_SECONDS, "samples": self.samples,
                "start": self.initial, "sampled_peaks": self.peaks, "gpu_names": {uid: name for _, uid, name in self.devices},
                "errors": sorted(self.errors), "scope": "NVML whole physical devices including other tasks; "
                "system_used = host total minus available; process_tree_rss = sum of readable RSS including descendants "
                "(shared pages can count more than once). Sampled maxima are lower bounds."}


def run_suite(teacher, case, passes, listening_notes, settings, render, unload, revision, sampler=MemorySampler):
    chosen = [item for item in CASES if case in ("all", item["id"])]
    if not chosen or passes not in ("cold_and_warm", "current_state_once"):
        raise ValueError("Unknown benchmark case or pass selection")
    rows, clips, targets, names = [], [], [], []
    for item in chosen:
        cast, source = fixed_reference(item["reference"])
        labels = ["current_state"] if passes == "current_state_once" else ["after_plugin_unload", "immediate_repeat"]
        for label in labels:
            if label == "after_plugin_unload":
                unload()
            first_chunk = None
            def on_chunk(data):
                nonlocal first_chunk
                if first_chunk is None and data["waveform"].shape[-1] > 0:
                    first_chunk = time.perf_counter()
            with sampler() as memory:
                start = time.perf_counter()
                audio, _, info = render(teacher, item["target"], item["seed"], cast=cast,
                                        settings=settings, mode="line", live=False, on_audio_chunk=on_chunk)
                end = time.perf_counter()
            duration = audio["waveform"].shape[-1] / audio["sample_rate"]
            run_id = item["id"] + "/" + label
            clips.append(audio)
            targets.append(item["target"])
            names.append(run_id)
            rows.append({"id": run_id, "case": item["id"], "pass": label, "seed": item["seed"],
                         "text": item["target"], "reference": source, "audio_seconds": duration,
                         "elapsed_seconds": end - start, "rtf": (end - start) / duration,
                         "first_stream_pcm_seconds": None if first_chunk is None else first_chunk - start,
                         "memory": memory.report(), "render": info})
    report = {"suite": SUITE, "model_revision": revision, "python": platform.python_version(),
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda, "platform": platform.platform(),
              "settings": settings, "runs": rows, "listening_notes": listening_notes, "asr_errors": None,
              "protocol": "Elapsed covers the render call. First PCM is the first nonempty streaming callback, "
              "null when none was emitted. Cold unloads this plugin's model before the call; OS disk caches and GPU "
              "driver remain warm. Immediate repeat uses the same reference, text, settings and seed. "
              "ASR assessment uses the optional matching AUDIO and target_text list outputs."}
    return clips, targets, names, json.dumps(report, ensure_ascii=False, indent=2)


class JRDrama3Benchmark(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 中英固定基准", category="audio/JR Drama 3/基准",
                         description="固定项目内合成参考、台本及种子。冷/热模式为每条台词各运行两次。"
                                     "clips 和 target_text 同序，可同时接本地 ASR 质检。",
                         inputs=[io.Custom("JR_DRAMA3_TEACHER").Input("teacher"),
                                 io.Combo.Input("case", options=["all", "zh", "en", "mixed"]),
                                 io.Combo.Input("passes", options=["cold_and_warm", "current_state_once"]),
                                 io.String.Input("listening_notes", default="", multiline=True),
                                 io.Custom("JR_DRAMA3_SAMPLING").Input("settings", optional=True)],
                         outputs=[io.Audio.Output("clips", is_output_list=True), io.String.Output("target_text", is_output_list=True),
                                  io.String.Output("run_id", is_output_list=True), io.String.Output("report_json")])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, teacher, case, passes, listening_notes, settings=None):
        from .jr_generate import DEFAULTS, render
        from .jr_gguf import QuantizedModel
        from .jr_teacher import REVISION
        revision = REVISION
        if isinstance(teacher, QuantizedModel):
            from functools import partial
            from .jr_realtime_client import unload_model
            unload = partial(unload_model, teacher.id, if_active=True)
            revision = teacher.id + "@" + teacher.sha256
        else:
            from .jr_runtime import unload
        return io.NodeOutput(*run_suite(teacher, case, passes, listening_notes, {**DEFAULTS, **(settings or {})},
                                        render, unload, revision))


def finalize_report(raw, review, notes, directory):
    report = json.loads(raw)
    if report.get("suite") != SUITE or not isinstance(report.get("runs"), list):
        raise ValueError("Expected a JR Drama 3 bilingual benchmark report")
    if review:
        if len(review) != len(report["runs"]):
            raise ValueError("Connect one readback result for each benchmark clip in its original order")
        for item, result in zip(report["runs"], review):
            item["readback_review"] = json.loads(result)
        report["asr_errors"] = "see runs[].readback_review"
    if notes.strip():
        report["listening_notes"] = notes.strip()
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(output.encode()).hexdigest() + ".json")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".tmp",
                                     delete=False) as temp:
        path = Path(temp.name)
        try:
            temp.write(output)
            temp.flush()
            os.fsync(temp.fileno())
        except BaseException:
            temp.close()
            path.unlink(missing_ok=True)
            raise
    try:
        os.replace(path, target)
    finally:
        path.unlink(missing_ok=True)
    return output, str(target)


class JRDrama3BenchmarkReport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Drama 3 · 归档基准报告", category="audio/JR Drama 3/基准",
                         is_input_list=True, is_output_node=True,
                         description="保存性能测量、可选逐条 ASR 差错及人工听审笔记到 ComfyUI/output/JR-Drama3/benchmarks。",
                         inputs=[io.String.Input("report_json", multiline=True), io.String.Input("listening_notes", multiline=True, default=""),
                                 io.String.Input("reviews_json", optional=True, force_input=True)],
                         outputs=[io.String.Output("full_report"), io.String.Output("report_file")])

    @classmethod
    def execute(cls, report_json, listening_notes, reviews_json=None):
        from folder_paths import get_output_directory
        from comfy_api.latest import ui
        if len(report_json) != 1:
            raise ValueError("Archive one complete benchmark report per node")
        out, file = finalize_report(report_json[0], reviews_json, listening_notes[0],
                                    Path(get_output_directory()) / "JR-Drama3" / "benchmarks")
        return io.NodeOutput(out, file, ui=ui.PreviewText(file))


NODES = [JRDrama3Benchmark, JRDrama3BenchmarkReport]
