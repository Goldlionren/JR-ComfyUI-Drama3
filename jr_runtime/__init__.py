"""Offline Fish Drama 3 FP8 teacher API. Built with Fish Audio.

Refs are {"speaker": 0..15, "text": transcript, "audio": Comfy AUDIO} or
pre-encoded {"speaker", "text", "codes"}. History is JSON-serializable.
The infill text argument is the entire corrected transcript. Previous audio
outside the edit is preserved at the mono 44,100 Hz working sample rate.
"""
import gc
import math
from pathlib import Path
import threading

import torch

from .attention import select_attention
from .inference import HOP, RATE, UnfinishedSpeechError, audio_dict, decode, encode, mono_audio, sample
from .model import load_codec, load_teacher, make_room_for_teacher
from .prompt import INFILL_INSTRUCTION, SCENE_INSTRUCTION, build_prompt, checked, codes_tensor, parse_scene, render_line, split_line


_lock = threading.RLock()
_active = None
MODES = ("line", "design", "scene", "continue", "revise", "infill", "stream")


def unload():
    """Release the teacher and its caches. A running synthesis completes first."""
    global _active
    with _lock:
        _active = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_engine(directory, device, cancel):
    global _active
    key = (Path(directory).resolve(), torch.device(device))
    if _active is None or _active[0] != key:
        unload()
        make_room_for_teacher(key[1])
        codec = load_codec(*key)
        model, offloaded = load_teacher(*key, cancel=cancel)
        _active = (key, model, codec, offloaded)
    return _active[1:]


def _encoded_refs(codec, refs, device):
    result = []
    for ref in refs:
        speaker = int(ref.get("speaker", 0))
        if not 0 <= speaker <= 15:
            raise ValueError("Speaker ID must be in 0–15.")
        transcript = checked(ref["text"], "reference transcript")
        if "codes" in ref:
            codes = codes_tensor(ref["codes"])
        else:
            codes = encode(codec, mono_audio(ref["audio"]), device)
        result.append({"speaker": speaker, "text": transcript, "codes": codes.tolist()})
    return result


class Preview:
    """Causal codec preview blocks. Retain left context when decoding a window."""
    def __init__(self, callback, codec, device):
        self.callback = callback
        self.codec = codec
        self.device = device
        self.last = None
        self.sent = 0

    def finish(self):
        if self.last is not None:
            self.send(self.last)
        self.last, self.sent = None, 0

    def send(self, codes):
        frames = codes.shape[1]
        if frames <= self.sent:
            return
        start = max(0, self.sent - 128)
        wave = decode(self.codec, codes[:, start:], self.device)
        self.callback(audio_dict(wave[(self.sent - start) * HOP:]))
        self.sent = frames

    def step(self, codes, is_separator):
        if is_separator:
            self.finish()
        else:
            self.last = codes
            if codes.shape[1] - self.sent >= 48:
                self.send(codes)


def _edit_inputs(codec, device, text, infill):
    if infill is None:
        raise ValueError("Infill requires source audio and start_seconds/end_seconds.")
    source = mono_audio(infill["audio"])
    times = (float(infill["start_seconds"]), float(infill["end_seconds"]))
    if not all(math.isfinite(x) for x in times):
        raise ValueError("Edit boundaries must be finite.")
    left, right = (round(x * RATE) for x in times)
    if not 0 <= left <= right <= source.numel():
        raise ValueError("Edit start/end are outside the source recording.")
    # Use frames wholly outside the edit as conditioning to avoid leaking
    # the old word through a partially overlapping frame. Retain up to 1,024
    # frames (~48s) on either side within the 8,192-row teacher prompt.
    before_end = (left // HOP) * HOP
    after_start = ((right + HOP - 1) // HOP) * HOP
    empty = torch.empty(10, 0, dtype=torch.long)
    prefix = source[max(0, before_end - 1024 * HOP):before_end]
    suffix = source[after_start:after_start + 1024 * HOP]
    before = encode(codec, prefix, device) if prefix.numel() else empty
    after = encode(codec, suffix, device) if suffix.numel() else empty
    if not before.numel() and not after.numel():
        raise ValueError("Infill needs at least one unchanged codec frame before or after the edit.")
    target = render_line(text, "")
    hint = infill.get("duration_frames")
    if hint is not None:
        if not 1 <= int(hint) <= 2048:
            raise ValueError("duration_frames must be in 1–2048.")
        target += f" [audio tokens: {int(hint)}]"
    fields = [target + "\n", "<|fim_suffix|>"]
    if after.numel():
        fields.append(after)
    fields.append("<|fim_prefix|>")
    if before.numel():
        fields.append(before)
    return fields, (source, left, right, before)


@torch.inference_mode()
def synthesize(model_dir, text, refs=(), seed=0, *, direction="", mode="line", previous=None,
               infill=None, temperature=0.7, top_p=0.9, top_k=0, max_frames=1024,
               progress=None, stream=None, cancel=None, device="cuda:0", use_fla=False):
    """Generate one take. Progress counts codec frames and stream emits mono AUDIO.

    A scene accepts JSON {scene,characters,lines}; observed model separators
    determine the returned segments. Incomplete sampling raises
    UnfinishedSpeechError. A valid deletion may finish as 'empty_infill'.
    """
    if mode not in MODES:
        raise ValueError(f"Unsupported teacher mode {mode!r}.")
    if mode not in ("continue", "revise", "stream"):
        previous = None
    if mode in ("continue", "revise") and not previous:
        raise ValueError(f"{mode} requires the history of an earlier take.")
    if not (1 <= int(max_frames) <= 8192 and math.isfinite(temperature) and temperature > 0
            and 0 < top_p <= 1 and 0 <= int(top_k) <= 4097):
        raise ValueError("Invalid sampling values or max_frames.")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The official FP8 teacher needs an NVIDIA CUDA GPU.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if previous is not None and previous.get("schema") != 1:
        raise ValueError("Unrecognized take history.")
    with _lock, torch.cuda.device(device), torch.random.fork_rng(devices=[device.index]):
        if cancel:
            cancel()
        torch.default_generator.manual_seed(int(seed))
        torch.cuda.default_generators[device.index].manual_seed(int(seed))
        model, codec, offloaded = _get_engine(model_dir, device, cancel)
        select_attention(model, use_fla)
        # Codec constructors initialize temporary weights. Start the actual
        # sampling sequence at the same seed on a cold and a warm load.
        torch.cuda.default_generators[device.index].manual_seed(int(seed))
        saved_refs = (previous or {}).get("references", [])
        references = _encoded_refs(codec, refs or saved_refs, device)
        turns = list((previous or {}).get("turns", []))
        system = ("convert the provided text to speech reference to the following:"
                  if references else "convert the provided text to speech")
        labels, edit = [], None
        if mode == "scene":
            system, script, labels = parse_scene(text, references, direction)
            jobs = [[script]]
        elif mode == "design":
            description = checked(direction, "voice description")
            system = f"Characters:\n- Narrator: {description}\n{SCENE_INSTRUCTION}"
            jobs = [[f"Narrator: {checked(text, 'text')}"]]
        elif mode == "stream":
            current = checked(text["current"], "current text chunk")
            lookahead = text.get("lookahead", "")
            if lookahead:
                lookahead = checked(lookahead, "next text chunk")
            if direction:
                current = render_line(current, direction)
            system = "streaming speech continuation with one-chunk lookahead split by <|endofturn|>"
            jobs = [[f"{current}<|endofturn|>{lookahead}"]]
        elif mode == "infill":
            fields, edit = _edit_inputs(codec, device, text, infill)
            jobs, system, turns, references = [fields], INFILL_INSTRUCTION, [], []
        else:
            jobs = [[render_line(chunk, direction)] for chunk in split_line(text)]
            if not jobs:
                raise ValueError("台词中只有角色标记，请填写要朗读的内容。")

        chunks, boundaries, reasons = [], [], []
        produced, cursor, edit_metadata = 0, 0, None
        scene_corrections = []
        for index, job in enumerate(jobs, 1):
            # Retain the opening voice and most recent performances within
            # the trained context, dropping the oldest intermediate take first.
            history = turns if len(turns) <= 3 else [turns[0], *turns[-2:]]
            history = list(history)
            while True:
                prompt = build_prompt(model, system, references, history, job)
                if len(prompt) <= 8192 or not history:
                    break
                history.pop(1 if len(history) > 1 else 0)
            preview = Preview(stream, codec, device) if stream and edit is None else None
            def on_frame(count, codes, boundary):
                if progress:
                    progress(produced + count, len(jobs) * max_frames)
                if preview is not None:
                    preview.step(codes, boundary)
            parts, reason, controls, corrections = sample(model, prompt, int(max_frames), float(temperature), float(top_p), int(top_k),
                                   max(0, len(labels) - 1), cancel, on_frame,
                                   allow_empty=bool(edit and edit[2] > edit[1]), scene_mode=bool(labels))
            scene_corrections.extend(corrections)
            if labels and (reason != "eos" or len(parts) != len(labels)):
                raise UnfinishedSpeechError(f"场景未完整结束（{reason}），共 {len(labels)} 句，"
                                            f"已生成 {len(parts)} 段。模型边界 {controls!r}。")
            if reason != "eos" and not (edit and reason == "empty_infill"):
                raise UnfinishedSpeechError(f"第 {index}/{len(jobs)} 段未完整结束（{reason}，"
                                            f"max_frames={max_frames}）。请缩短台词、提高帧数或更换种子。")
            if preview:
                preview.finish()
            reasons.append(reason)
            produced += sum(part.shape[1] for part in parts)
            for i, part in enumerate(parts):
                if edit:
                    source, left, right, before = edit
                    prefix = before[:, -128:]
                    if part.shape[1]:
                        codes = torch.cat((prefix, part), dim=1)
                        middle = decode(codec, codes, device)[prefix.shape[1] * HOP:]
                    else:
                        middle = source.new_empty(0)
                    wave = torch.cat((source[:left], middle, source[right:]))
                    edit_metadata = {"start_sample": left, "original_end_sample": right,
                                     "new_end_sample": left + middle.numel()}
                else:
                    wave = decode(codec, part, device)
                chunks.append(wave)
                label = labels[i] if labels and i < len(labels) else {"text": text if edit else job[0]}
                boundaries.append({**label, "start_sample": cursor, "end_sample": cursor + wave.numel()})
                cursor += wave.numel()
            if edit:
                # Subsequent revisions condition on the *whole edited take*.
                turns.append({"user": render_line(text, ""), "parts": [encode(codec, wave, device).tolist()]})
            else:
                turns.append({"user": job[0], "parts": [p.tolist() for p in parts], "controls": controls})
            # Match the next-turn context and bound per-take checkpoint size
            # over long books. Older, unbounded schema-1 histories still load.
            if len(turns) > 3:
                turns = [turns[0], *turns[-2:]]
        wave = audio_dict(torch.cat(chunks))
        return {"audio": wave, "frames": produced, "seed": int(seed),
                "finish_reason": next((r for r in reasons if r != "eos"), "eos"),
                "history": {"schema": 1, "references": references, "turns": turns},
                "segments": boundaries, "edit": edit_metadata, "scene_corrections": scene_corrections,
                "host_expert_layers": offloaded}


def synthesize_stream(model_dir, chunks, refs=(), seed=0, *, previous=None, direction="",
                      temperature=0.7, top_p=0.9, top_k=0, max_frames=1024,
                      progress=None, stream=None, cancel=None, on_take=None, device="cuda:0", use_fla=False):
    """Consume a live iterable of text using figure 10's one-chunk lookahead.

    The caller's iterator may block for input and should poll cancel() while
    waiting. stream(AUDIO) emits incremental PCM. on_take(index, result) gets
    the completed standalone result for each text chunk. Returns a combined
    result with sample-accurate per-chunk spans and the final JSON history.
    """
    if isinstance(chunks, (str, bytes)):
        raise ValueError("chunks must be an iterable of text chunks, not one string.")
    def next_text(iterator):
        if cancel:
            cancel()
        for item in iterator:
            if not isinstance(item, str):
                raise ValueError("Live text chunks must be strings.")
            if item.strip():
                return item.strip()
            if cancel:
                cancel()
        return None
    iterator = iter(chunks)
    current = next_text(iterator)
    if current is None:
        raise ValueError("No text was submitted to the live session.")
    audio, segments = [], []
    history = previous
    cursor = frames = 0
    while current is not None:
        following = next_text(iterator)
        index = len(segments)
        result = synthesize(model_dir, {"current": current, "lookahead": following or ""},
                            refs if index == 0 else (), (int(seed) + index) % (2**63),
                            mode="stream", previous=history, direction=direction, max_frames=max_frames,
                            temperature=temperature, top_p=top_p, top_k=top_k, progress=progress,
                            stream=stream, cancel=cancel, device=device, use_fla=use_fla)
        if result["finish_reason"] != "eos":
            raise UnfinishedSpeechError(f"实时台词第 {index + 1} 段未完整结束（{result['finish_reason']}）。")
        clip = result["audio"]["waveform"]
        audio.append(clip)
        end = cursor + clip.shape[-1]
        segments.append({"index": index, "text": current, "start_sample": cursor, "end_sample": end,
                         "finish_reason": result["finish_reason"]})
        history, cursor = result["history"], end
        frames += result["frames"]
        if on_take:
            on_take(index, result)
        current = following
    return {"audio": audio_dict(torch.cat(audio, dim=-1)), "history": history, "seed": int(seed),
            "segments": segments, "frames": frames, "finish_reason": result["finish_reason"],
            "host_expert_layers": result["host_expert_layers"]}
