"""Fish Drama 3 prompt layouts from the technical report, figures 11–14."""
import json
import re

import torch

if "." in __package__:
    from ..jr_script_core import PHONEME, chunk_script
else:  # Also used by the standalone local teacher API and its tests.
    from jr_script_core import PHONEME, chunk_script


SEPARATOR = "<|audio_separator|>"
END = "<|im_end|>\n"
START = "<|im_start|>assistant\n<|voice|>"
SPEAKER = re.compile(r"<\|speaker:(\d+)\|>")
PHONEMES = re.compile(PHONEME)
SCENE_INSTRUCTION = "Voice every line of the script in order, with <|audio_separator|> between consecutive lines."
INFILL_INSTRUCTION = "fill in the missing part of the speech; the suffix follows <|fim_suffix|> and the prefix follows <|fim_prefix|>"


def checked(value, field):
    from fish_speech.models.text2semantic.fish_omni.generate import validate_drama3_text
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected text.")
    value = value.strip()
    validate_drama3_text(value, field)
    if "<|" in SPEAKER.sub("", PHONEMES.sub("", value)):
        raise ValueError(f"{field}: reserved model control token")
    return value


def split_line(text):
    from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text
    value = checked(text, "text")
    if PHONEMES.search(value) and len(value.encode("utf-8")) > 300:
        # Upstream protects each delimiter separately. Keep complete pairs
        # together so the next chunk never sees a dangling pronunciation span.
        return chunk_script(value, 300)[0]
    return split_drama3_text(value)


def verify_control_tokens(tokenizer):
    for marker in ("<|phoneme_start|>", "<|phoneme_end|>", SEPARATOR, "<|fim_prefix|>",
                   "<|fim_suffix|>", "<|endofturn|>"):
        token_id = tokenizer.token_to_id(marker)
        if token_id is None or tokenizer.encode(marker, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f"Teacher tokenizer does not encode {marker} as one control token.")


def render_line(text, direction):
    from fish_speech.models.text2semantic.fish_omni.generate import with_speaker_tag
    text = with_speaker_tag(checked(text, "text"))
    if not direction:
        return text
    direction = checked(direction, "direction")
    tag = SPEAKER.match(text.lstrip())
    return f"{tag.group(0)} Direction: {direction}\nLine: {text.lstrip()[tag.end():].lstrip()}"


def parse_scene(text, refs, direction):
    scene = json.loads(text) if isinstance(text, str) else text
    lines = scene["lines"]
    if not lines:
        raise ValueError("Scene has no script lines.")
    cast = {}
    for character in scene.get("characters", []):
        speaker = int(character["speaker"])
        if not 0 <= speaker <= 15 or speaker in cast:
            raise ValueError("Scene character IDs must be unique integers in 0–15.")
        name = checked(character["name"], "character name")
        description = character.get("description") or ""
        cast[speaker] = (name, checked(description, "description") if description else "")
    parts, labels = [], []
    for line in lines:
        speaker = int(line["speaker"])
        if not 0 <= speaker <= 15:
            raise ValueError("Scene speaker IDs must be in 0–15.")
        name = checked(line.get("name") or cast.get(speaker, (str(speaker), ""))[0], "speaker name")
        script = checked(line["text"], "line")
        # Keep names out of the spoken text. The fixed speaker tokens carry
        # the cast mapping even when voices are designed without references.
        parts.append(f"<|speaker:{speaker}|> {script}")
        labels.append({"speaker": speaker, "name": name, "text": script})
    context = []
    if scene.get("scene"):
        context.append(f"Scene: {checked(scene['scene'], 'scene')}")
    if direction:
        context.append(f"Direction: {checked(direction, 'direction')}")
    if cast:
        context.append("Characters:")
        for speaker, (name, description) in cast.items():
            display = f"<|speaker:{speaker}|> {name}"
            context.append(f"- {display}: {description}")
    context.append(SCENE_INSTRUCTION)
    return "\n".join(context), ("\n" + SEPARATOR + "\n").join(parts), labels


def codes_tensor(value):
    tensor = torch.as_tensor(value)
    if (tensor.ndim != 2 or tensor.shape[0] != 10 or tensor.shape[1] == 0
            or tensor.dtype.is_floating_point or tensor.dtype == torch.bool
            or tensor.min() < 0 or tensor.max() >= 4096):
        raise ValueError("Reference VQ codes must be integers with shape [10,frames] and values 0–4095.")
    return tensor.to(device="cpu", dtype=torch.long)


def build_prompt(model, system, refs, turns, user_parts):
    fields = ["<|im_start|>system\n", system]
    for i, ref in enumerate(refs, start=1):
        transcript = checked(ref["text"], "reference transcript")
        speaker = int(ref["speaker"])
        if not 0 <= speaker <= 15:
            raise ValueError("Reference speaker ID must be in 0–15.")
        fields.extend([f"\n\n[Reference {i}]\nText: <|speaker:{speaker}|> {transcript}\nSpeech: ",
                       codes_tensor(ref["codes"])])
    fields.append(END)
    for turn in turns:
        fields.extend(["<|im_start|>user\n", turn["user"], END, START])
        if turn.get("controls"):
            audio = torch.cat([codes_tensor(part) for part in turn["parts"]], dim=1)
            start = 0
            for offset, marker in turn["controls"]:
                if not start <= offset <= audio.shape[1] or marker not in ("\n", SEPARATOR):
                    raise ValueError("Invalid scene token history.")
                if offset > start:
                    fields.append(audio[:, start:offset])
                fields.append(marker)
                start = offset
            if start < audio.shape[1]:
                fields.append(audio[:, start:])
        else:
            for i, part in enumerate(turn["parts"]):
                if i:
                    fields.append(SEPARATOR)
                fields.append(codes_tensor(part))
        fields.append(END)
    fields.extend(["<|im_start|>user\n", *user_parts, END, START])
    rows = []
    for field in fields:
        if isinstance(field, str):
            ids = model.tokenizer.encode(field).ids
            block = torch.zeros((len(ids), 11), dtype=torch.long)
            block[:, 0] = torch.as_tensor(ids)
        else:
            semantic = field[:1] + model.config.semantic_start_token_id
            block = torch.cat([semantic, field]).T
        rows.append(block)
    return torch.cat(rows)
