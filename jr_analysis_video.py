"""Original-frame dubbing, offline subtitle translation and local stem separation."""

from __future__ import annotations

from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import soundfile as sf

from .jr_analysis_asr import AUX, LOCAL_GPT, ordered_segments, source_path
from .jr_delivery_audio import as_audio, convert, resample, samples


def subtitle_text(segs):
    def stamp(seconds):
        t = round(seconds * 1000)
        return f"{t // 3600000:02}:{t // 60000 % 60:02}:{t // 1000 % 60:02},{t % 1000:03}"

    return (
        "\n\n".join(
            f"{i}\n{stamp(s['start'])} --> {stamp(s['end'])}\n{s['text']}" for i, s in enumerate(segs, 1)
        )
        + "\n"
    )


def dubbing_script(
    segments,
    translations="",
    model_dir="",
    src_lang="eng_Latn",
    tgt_lang="zho_Hans",
    translation_mode="manual",
):
    segs = ordered_segments(segments)
    named = {"en → zh": "opus-mt-en-zh", "zh → en": "opus-mt-zh-en"}
    if translation_mode not in ("manual", *named):
        raise ValueError("Unknown translation mode.")
    if translations.strip() and (model_dir.strip() or translation_mode in named):
        raise ValueError("Choose a manual translation or a local translation model.")
    texts = [s["text"] for s in segs]
    if translations.strip():
        out = [line.strip() for line in translations.strip().splitlines()]
        if len(out) != len(segs) or not all(out):
            raise ValueError("Enter one nonempty translated line per segment, in order.")
    elif model_dir.strip() or translation_mode in named:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if translation_mode in named:
            model_name = named[translation_mode]
            path = source_path(
                model_dir,
                "JR_DRAMA3_" + model_name.upper().replace("-", "_") + "_DIR",
                [AUX / model_name],
                directory=True,
            )
        else:
            path = source_path(model_dir, "JR_DRAMA3_NLLB_DIR", [], directory=True)
        family = json.loads((path / "config.json").read_text("utf-8"))["model_type"]
        marian = family == "marian"
        options = {} if marian else {"src_lang": src_lang}
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, **options)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        ).eval()
        generate = {}
        if not marian:
            lang_id = tok.convert_tokens_to_ids(tgt_lang)
            if lang_id is None or lang_id == tok.unk_token_id:
                raise ValueError("Target language is missing from this local NLLB tokenizer.")
            generate["forced_bos_token_id"] = lang_id
        out = []
        for text in texts:
            batch = tok(text, return_tensors="pt")
            if batch["input_ids"].shape[-1] > model.config.max_position_embeddings:
                raise ValueError(
                    "Subtitle exceeds the translation model context; divide it at a time boundary."
                )
            tokens = model.generate(
                **batch, **generate, max_length=min(512, model.config.max_position_embeddings)
            )
            out.append(tok.decode(tokens[0], skip_special_tokens=True))
    else:
        out = texts
    for s, translated in zip(segs, out):
        s["source_text"] = s["text"]
        s["text"] = translated
    script = "\n".join(f"{s['speaker']}: {s['text']}" if s.get("speaker") else s["text"] for s in segs)
    return script, subtitle_text(segs), segs


@contextmanager
def stream_source(video):
    """Use the effective trimmed/cropped view; source paths can refer to a larger file."""
    trim = getattr(video, "get_active_trim_window", lambda: (0, 0))()
    crop = getattr(video, "get_active_crop_rect", lambda: None)()
    if crop or any(trim):
        with tempfile.TemporaryDirectory(prefix="jr-drama3-view-") as tmp:
            p = Path(tmp) / "source.mp4"
            video.save_to(str(p))
            yield p
        return
    source = video.get_stream_source()
    if isinstance(source, (str, os.PathLike)):
        yield source_path(source, "JR_DRAMA3_VIDEO", [])
        return
    if not hasattr(source, "read"):
        raise ValueError("Expected a local encoded video stream.")
    with tempfile.TemporaryDirectory(prefix="jr-drama3-stream-") as tmp:
        p = Path(tmp) / "source.video"
        pos = source.tell()
        try:
            source.seek(0)
            with p.open("wb") as out:
                while block := source.read(1 << 20):
                    out.write(block)
        finally:
            source.seek(pos)
        yield p


def extract_track(video):
    with stream_source(video) as p:
        raw = convert(
            [
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                p,
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "pcm_f32le",
                "-f",
                "wav",
                "pipe:1",
            ]
        )
    x, rate = sf.read(io.BytesIO(raw), always_2d=True, dtype="float32")
    return as_audio(x, rate)


def premix(dub, background=None, foreground_db=0, background_db=0, offset=0):
    voice, sr = samples(dub)
    if not all(np.isfinite([foreground_db, background_db, offset])) or offset < 0:
        raise ValueError("Use finite gains and a nonnegative start offset.")
    lead = round(sr * offset)
    if background is not None:
        bed, br = samples(background)
        bed = resample(bed, br, sr)
        channels = max(voice.shape[1], bed.shape[1])
        size = max(len(voice) + lead, len(bed))
    else:
        channels = voice.shape[1]
        size = len(voice) + lead
    out = np.zeros((size, channels), dtype=np.float32)
    out[lead : lead + len(voice)] += voice * 10 ** (foreground_db / 20)
    if background is not None:
        out[: len(bed)] += bed * 10 ** (background_db / 20)
    peak = float(np.max(np.abs(out)))
    scale = 1 / max(1.0, peak)
    return as_audio(out * scale, sr), {
        "sample_rate": sr,
        "samples": size,
        "channels": channels,
        "peak_scaling": scale,
        "start": offset,
    }


def h3_timing(audio):
    x, rate = samples(audio)
    requested = math.ceil(len(x) * 24 / rate)
    frames = 5 + 17 * max(0, math.ceil((requested - 5) / 17))
    if frames > 3600:
        raise ValueError("Split the recording into shots within H3's 3600-frame input limit.")
    target = round(frames * rate / 24)
    if target > len(x):
        x = np.pad(x, ((0, target - len(x)), (0, 0)))
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return as_audio(x, rate), frames


def remux(video, audio, output_dir, prefix="jr-drama3-dub"):
    import av

    x, rate = samples(audio)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not prefix or Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("Expected a simple output file name.")
    output = out_dir / f"{prefix}-{uuid.uuid4().hex}.mkv"
    try:
        with stream_source(video) as source, tempfile.TemporaryDirectory(prefix="jr-drama3-mux-") as tmp:
            # An explicit endpoint prevents the last video packet's DTS (B frames) shortening the soundtrack.
            with av.open(str(source), options={"protocol_whitelist": "file,pipe"}) as container:
                vs = container.streams.video[0]
                if vs.duration is not None:
                    duration = float(vs.duration * vs.time_base)
                else:
                    origin = None
                    end = 0.0
                    for packet in container.demux(vs):
                        if packet.pts is None:
                            continue
                        ts = float(packet.pts * packet.time_base)
                        origin = ts if origin is None else min(ts, origin)
                        end = max(end, ts + float((packet.duration or 0) * packet.time_base))
                    duration = end - (origin or 0.0)
            if duration <= 0:
                raise ValueError("Could not determine original video duration.")
            wave = Path(tmp) / "mix.wav"
            sf.write(wave, x, rate, subtype="FLOAT")
            convert(
                [
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    source,
                    "-i",
                    wave,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "flac",
                    "-af",
                    f"apad,atrim=duration={duration:.9f}",
                    "-y",
                    output,
                ]
            )
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def separate_stems(audio, device="cpu"):
    runner = Path(__file__).with_name("jr_analysis_roformer.py")
    app = source_path("", "JR_DRAMA3_GPT_SOVITS_DIR", [LOCAL_GPT], directory=True)
    model = source_path(
        "",
        "JR_DRAMA3_BSROFORMER_PATH",
        [app / "tools/uvr5/uvr5_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt"],
    )
    x, sr = samples(audio)
    with tempfile.TemporaryDirectory(prefix="jr-drama3-stems-") as tmp:
        tmp = Path(tmp)
        sf.write(tmp / "in.wav", x, sr, subtype="FLOAT")
        params = [sys.executable, "-B", str(runner), str(app / "tools/uvr5"), str(model), str(tmp), device]
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        result = subprocess.run(
            params,
            cwd=tmp,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf8", "replace")[-1800:])
        voice, rate = sf.read(tmp / "vocal.wav", always_2d=True, dtype="float32")
    # The model operates at 44.1 kHz stereo. Compute the residual at native precision.
    original = resample(x, sr, rate)
    return as_audio(voice, rate), as_audio(original[: len(voice)] - voice, rate)
