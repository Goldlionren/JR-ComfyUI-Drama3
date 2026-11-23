"""Local autoregressive sampling, streaming preview and codec conversion."""
import math

import torch

from .prompt import SEPARATOR


RATE = 44100
HOP = 2048


class UnfinishedSpeechError(RuntimeError):
    """A recoverable generation failure for this text, seed or frame budget."""


def audio_dict(waveform):
    mono = waveform.detach().to(device="cpu", dtype=torch.float32).reshape(1, 1, -1).contiguous()
    if mono.numel() == 0 or not torch.isfinite(mono).all():
        raise ValueError("Teacher produced empty or non-finite audio.")
    return {"waveform": mono, "sample_rate": RATE}


def mono_audio(audio):
    source = torch.as_tensor(audio["waveform"]).detach().to(device="cpu", dtype=torch.float32)
    rate = int(audio["sample_rate"])
    if source.ndim != 3 or source.shape[0] != 1 or source.shape[1] < 1 or source.shape[2] < 1 or rate <= 0:
        raise ValueError("Expected one Comfy AUDIO recording with shape [1,channels,samples].")
    source = source[0].mean(dim=0)
    if not torch.isfinite(source).all():
        raise ValueError("Audio contains non-finite samples.")
    if rate != RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(rate, RATE)
        source = torch.from_numpy(resample_poly(source.numpy(), RATE // divisor, rate // divisor))
    return source.contiguous()


def encode(codec, wave, device):
    result = []
    for first in range(0, wave.numel(), 256 * HOP):
        start = max(0, first - 128 * HOP)
        window = wave[start:first + 256 * HOP]
        value = window.to(device=device, dtype=next(codec.parameters()).dtype).reshape(1, 1, -1)
        ids, lengths = codec.encode(value, torch.tensor([window.numel()], device=device))
        result.append(ids[0, :, (first - start) // HOP:int(lengths[0])].long().cpu())
    return torch.cat(result, dim=1)


def decode(codec, codes, device):
    result = []
    for first in range(0, codes.shape[1], 256):
        start = max(0, first - 128)
        window = codes[:, start:first + 256]
        out = codec.from_indices(window[None].to(device))[0, 0]
        result.append(out[(first - start) * HOP:].float().cpu().contiguous())
    return torch.cat(result)


def scene_frame(tower, logits, hidden, position, previous_frame, previous_ids, *,
                temperature, top_p, top_k, newline, early_eos):
    """Choose a legal scene token before deciding whether to run the codebooks.

    Both newline and the line separator are non-acoustic teacher outputs.
    Their temporal hidden states advance, but no audio codebooks are sampled.
    """
    from fish_speech.models.text2semantic.fish_omni.tower import (
        RAS_WINDOW_FRAMES, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P,
        RESIDUAL_CODEBOOK_SIZE, reference_sample,
    )
    candidates = tower.main_candidates
    scores = logits[:, candidates]
    selected = candidates[reference_sample(scores, temperature, top_k, top_p, 0)]
    recent = previous_ids[-RAS_WINDOW_FRAMES:]
    if bool(tower.is_semantic(selected).all() and (selected[:, None] == recent).any()):
        selected = candidates[reference_sample(scores, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, 0)]
    token = int(selected.item())
    corrected = token == tower.eos_id and early_eos
    if corrected:
        token = newline
        selected.fill_(token)
    row = torch.zeros_like(previous_frame)
    row[:, 0] = selected
    if token == tower.eos_id:
        return row, token, corrected
    if not tower.semantic_begin <= token <= tower.semantic_end:
        tower.prefill(hidden, position, previous_frame, row)
        return row, token, corrected
    code = selected - tower.semantic_begin
    row[:, 1] = code
    semantic = tower.is_semantic(selected)
    state = tower.run_shared(hidden, code, semantic, position, previous_frame)
    for i, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads), start=2):
        state = layer(state, code, semantic, position)
        code = reference_sample(head(state), temperature, top_k, top_p, 0,
                                legal_column_count=RESIDUAL_CODEBOOK_SIZE)
        row[:, i] = code
    return row, token, corrected


def sample(model, prompt, max_frames, temperature, top_p, top_k, separators, cancel, on_frame, *,
           allow_empty=False, scene_mode=False):
    from fish_speech.models.text2semantic.fish_omni.generate import ensure_drama3_caches
    backbone, tower = model.backbone, model.audio_decoder
    separator = model.tokenizer.token_to_id(SEPARATOR)
    vocab = torch.arange(tower.semantic_begin, tower.semantic_end + 1, device=tower.device)
    special = [model.config.eos_token_id]
    newline = model.tokenizer.encode("\n", add_special_tokens=False).ids[0]
    if scene_mode:
        if separator is None:
            raise ValueError("Teacher tokenizer lacks the scene line separator.")
        special.append(newline)
        if separators:
            special.append(separator)
    tower.main_candidates = torch.cat([torch.tensor(special, device=tower.device), vocab]).sort().values
    control_capacity = 4 * (separators + 1) if scene_mode else 0
    available = 10240 - len(prompt) - control_capacity
    if len(prompt) > 8192 or available < 1:
        raise ValueError(f"Prompt exceeds teacher runtime limits ({len(prompt)} prompt rows).")
    context_limited = max_frames > available
    max_frames = min(max_frames, available)
    capacity = len(prompt) + max_frames + control_capacity
    ensure_drama3_caches(model, capacity)
    rows = torch.zeros((capacity, 11), dtype=torch.long, device=tower.device)
    rows[:len(prompt)] = prompt.to(tower.device)
    positions = torch.arange(capacity, device=tower.device)
    for first in range(0, len(prompt), 64):
        if cancel:
            cancel()
        end = min(first + 64, len(prompt))
        block = rows[first:end]
        embedding = tower.embed_backbone(block, backbone.embed(block[:, 0]))
        hidden = backbone(inputs_embeds=embedding[None], input_pos=positions[first:end])[0]
        stop = min(end, len(prompt) - 1)
        if first < stop:
            tower.prefill(hidden[:stop-first], positions[first:stop], rows[first:stop], rows[first+1:stop+1])
    hidden = hidden[-1:]
    audio = torch.empty((10, max_frames), dtype=torch.long, device=tower.device)
    pieces, controls, corrections, start_audio, spoken = [], [], [], 0, 0
    redirected = False
    position = len(prompt) - 1
    reason = "context_limit" if context_limited else "length"
    # At the exact frame limit the next model token may still be EOS. A
    # terminal/control token consumes no acoustic frame and fits the KV cache.
    while True:
        if cancel:
            cancel()
        logits = backbone.compute_logits(hidden)
        if scene_mode:
            frame, token, corrected = scene_frame(
                tower, logits, hidden, positions[position:position+1], rows[position:position+1], rows[:position+1, 0],
                temperature=temperature, top_p=top_p, top_k=top_k, newline=newline,
                early_eos=len(pieces) < separators and spoken > start_audio and not redirected)
            redirected |= corrected
            if corrected:
                corrections.append({"audio_frame": spoken, "line_index": len(pieces), "action": "early_eos_to_newline"})
        else:
            frame = tower.decode_step(logits, hidden, positions[position:position+1], rows[position:position+1],
                                      rows[:position+1, 0], temperature=temperature, top_p=top_p,
                                      top_k=top_k, epsilon_cutoff=0.0)
            token = int(frame[0, 0])
        if token == model.config.eos_token_id:
            reason = "eos"
            break
        if scene_mode and token in (newline, separator):
            if len(controls) >= control_capacity:
                reason = "control_limit"
                break
            controls.append([spoken, "\n" if token == newline else SEPARATOR])
            if token == separator:
                if spoken == start_audio or len(pieces) >= separators:
                    reason = "unexpected_separator"
                    break
                pieces.append(audio[:, start_audio:spoken].cpu().contiguous())
                start_audio, redirected = spoken, False
                on_frame(spoken, audio[:, spoken:spoken], True)
        else:
            if spoken == max_frames:
                break
            audio[:, spoken] = frame[0, 1:]
            spoken += 1
            on_frame(spoken, audio[:, start_audio:spoken], False)
        position += 1
        rows[position] = frame[0]
        embedding = tower.embed_backbone(frame, backbone.embed(frame[:, 0]))
        hidden = backbone(inputs_embeds=embedding[None], input_pos=positions[position:position+1])[0]
    if spoken > start_audio:
        pieces.append(audio[:, start_audio:spoken].cpu().contiguous())
    if not pieces:
        if allow_empty and reason == "eos":
            # A deleted span can legitimately consist of zero new frames.
            return [torch.empty((10, 0), dtype=torch.long)], "empty_infill", [], []
        raise UnfinishedSpeechError("教师在发声前结束。请更换种子或调整导演意见。")
    if reason == "eos" and len(pieces) != separators + 1:
        reason = "incomplete_scene"
    return pieces, reason, controls, corrections
