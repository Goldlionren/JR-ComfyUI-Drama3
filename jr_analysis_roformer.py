"""Child process for the locally installed GPT-SoVITS BS-RoFormer architecture."""

import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def main(uvr_dir, checkpoint, workdir, device):
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda for the separation model.")
    sys.path.insert(0, str(Path(uvr_dir).resolve()))
    from bsroformer import BsRoformer_Loader

    runner = BsRoformer_Loader.__new__(BsRoformer_Loader)
    model = runner.get_model_from_config()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to(device)
    root = Path(workdir)
    x, sr = sf.read(root / "in.wav", dtype="float32", always_2d=True)
    if sr != 44100:
        g = math.gcd(sr, 44100)
        x = resample_poly(x, 44100 // g, sr // g, axis=0)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    window = 8 * 44100
    step = window // 2
    mix = torch.from_numpy(np.ascontiguousarray(x.T))
    den = torch.zeros(mix.shape[-1])
    result = torch.zeros_like(mix)
    with torch.inference_mode():
        for start in range(0, mix.shape[-1], step):
            part = mix[:, start : start + window]
            n = part.shape[1]
            # Align STFT output length and leave room for the model's reflect padding.
            padded = math.ceil(max(n, 2048) / 441) * 441
            inp = torch.nn.functional.pad(part, (0, padded - n))
            pred = model(inp[None].to(device)).cpu().reshape(2, -1)[:, :n]
            if pred.shape[-1] != n:
                raise RuntimeError("The separation model produced a shortened segment.")
            gain = torch.hann_window(n + 2, periodic=False)[1:-1].clamp_min(0.001)
            result[:, start : start + n] += pred * gain
            den[start : start + n] += gain
            if start + n == mix.shape[-1]:
                break
    voice = (result / den).T.numpy()
    if not np.isfinite(voice).all():
        raise ValueError("Non-finite values from BS-RoFormer.")
    sf.write(root / "vocal.wav", voice, 44100, subtype="FLOAT")


if __name__ == "__main__":
    main(*sys.argv[1:])
