"""Build the standalone five-language audio.cpp package from the pinned TensorFold checkpoint."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile

import gguf
import numpy as np
from safetensors import safe_open
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
SOURCE = "ThinkingVibeCoder/Drama-3-Tier2-5Lang-INT3"
REVISION = "e3855f74beb7579fcfbe7aadccf9ee9ab330404e"
FILENAME = "Drama3-Tier2-5Lang.gguf"
Q = gguf.GGMLQuantizationType
V = gguf.GGUFValueType
E4M3 = np.array([((i & 7) * 2**-9 if i < 8 else (1 + (i & 7) / 8) * 2**((i >> 3) - 7))
                 for i in range(127)], dtype=np.float32)
# Spread each bit of a bit-plane byte into the low bit of a separate output byte.
SPREAD = np.array([sum(((x >> i) & 1) << (8 * i) for i in range(8)) for x in range(256)], dtype="<u8")


def repack_int3(weight, scales):
    """Losslessly map the source sign/hi/lo planes to GGML's 64-weight E2M1 blocks."""
    rows, cols = weight.shape
    planes = weight.reshape(rows, -1, 6)
    if scales.shape != planes.shape[:2] or np.any(scales >= 127):
        raise ValueError("Invalid INT3 block scales")
    blocks = np.empty((rows, cols // 24, 36), dtype=np.uint8)
    blocks[:, :, :4] = scales.reshape(rows, -1, 4)
    packed = np.zeros(scales.shape, dtype="<u8")
    for half in (0, 1):
        sign, hi, lo = (SPREAD[planes[:, :, p + half]] for p in (0, 2, 4))
        # E2M1 magnitudes 0, 2, 4, 6 have nibble indices 0, 4, 6, 7.
        nibble = (sign << 3) | ((hi | lo) << 2) | (hi << 1) | (hi & lo)
        packed |= nibble << (4 * half)
    blocks[:, :, 4:] = packed.view(np.uint8).reshape(rows, -1, 32)
    return blocks.reshape(rows, -1)


def compact_int3(weight, scales, global_scale):
    """Keep the source signed 3-bit coefficients. Fit only the GGML per-group scales."""
    if not np.isfinite(global_scale) or global_scale <= 0 or np.any(scales >= len(E4M3)):
        raise ValueError("Invalid source expert scales")
    rows = weight.shape[0]
    planes = weight.reshape(-1, 16, 3, 2)
    bits = np.unpackbits(planes, axis=-1, bitorder="little")
    sign, hi, lo = (bits[:, :, i].astype(np.int8) for i in range(3))
    codes = (2 * hi + lo) * (1 - 2 * sign)
    group = E4M3[scales.reshape(-1, 16)] * (2 * global_scale)
    energy = np.sum(codes * codes, axis=2, dtype=np.float32)
    peak = np.max(group, axis=1, keepdims=True)
    best_error = np.full_like(peak, np.inf)
    best_d = np.zeros_like(peak)
    best_levels = np.zeros_like(group, dtype=np.uint8)
    # E4M3 groups often share binary exponents. Include their exact binary grid
    # alongside a fitted linear grid with the full signed 6-bit scale range.
    binary = np.exp2(np.ceil(np.log2(np.maximum(peak, 1e-30) / global_scale / 32))) * global_scale
    for step in (binary, peak / 32, peak / 30):
        for iteration in range(2):
            step = np.maximum(step.astype(np.float16).astype(np.float32), 2**-24)
            levels = np.clip(np.rint(group / step), 0, 32)
            error = np.sum(energy * (group - levels * step)**2, axis=1, keepdims=True)
            better = error < best_error
            best_d = np.where(better, step, best_d)
            best_levels = np.where(better, levels.astype(np.uint8), best_levels)
            best_error = np.minimum(best_error, error)
            if iteration == 0:
                step = np.sum(energy * levels * group, axis=1, keepdims=True) / np.maximum(
                    np.sum(energy * levels**2, axis=1, keepdims=True), 1)
    out = np.empty((len(planes), 110), dtype=np.uint8)
    out[:, :32] = np.packbits((codes >= 0).reshape(-1, 8, 32), axis=1, bitorder="little")[:, 0]
    low = (codes.reshape(-1, 2, 4, 32).astype(np.uint8) & 3)
    out[:, 32:96] = np.bitwise_or.reduce(low << np.array([0, 2, 4, 6], np.uint8)[None, None, :, None],
                                           axis=2).reshape(-1, 64)
    s = 32 - best_levels
    out[:, 96:104] = (s[:, :8] & 15) | (s[:, 8:] << 4)
    out[:, 104:108] = np.bitwise_or.reduce((s.reshape(-1, 4, 4) >> 4) <<
                                           np.array([0, 2, 4, 6], np.uint8)[None, :, None], axis=1)
    out[:, 108:] = (-best_d).astype("<f2").view(np.uint8)
    return out.reshape(rows, -1)


def q8(values):
    # Very small source FP32 denormals have zero scale in Q8_0's FP16 header.
    # Clear those blocks before gguf computes their reciprocal in FP32.
    blocks = values.reshape(-1, 32)
    tiny = np.max(np.abs(blocks), axis=1) < (2**-25 * 127)
    if tiny.any():
        values = values.copy()
        values.reshape(-1, 32)[tiny] = 0
    return gguf.quantize(values, Q.Q8_0)


def verify_source(path):
    manifest = json.loads((ROOT / "resources/five-lang-source.json").read_text())
    for name, digest in manifest.items():
        file = path / name
        with file.open("rb") as stream:
            if len(digest) == 40:  # Hugging Face Git blob ID for small sidecars
                check = hashlib.sha1(f"blob {file.stat().st_size}\0".encode() + stream.read()).hexdigest()
            else:
                check = hashlib.file_digest(stream, "sha256").hexdigest()
        if check != digest:
            raise ValueError(f"Source checksum mismatch: {file}")


def build(source, output, spec, reference=False):
    verify_source(source)
    config = json.loads((source / "config.json").read_bytes())
    index = json.loads((source / "model.safetensors.index.json").read_bytes())
    weights = index["weight_map"]
    retained = [index["metadata"]["drama_experts"][str(i)] for i in range(40)]
    head = index["metadata"]["drama_head_rows"]
    if sum(map(len, retained)) != 6858 or head != [config["eos_token_id"], *range(
            config["semantic_start_token_id"], config["semantic_end_token_id"] + 1)]:
        raise ValueError("Unexpected five-language expert map or semantic head")
    config["text_config"]["num_experts_per_layer"] = list(map(len, retained))
    config["semantic_head_rows"] = len(head)
    writer = gguf.GGUFWriter(output, "audiocpp", use_temp_file=True)
    writer.add_name("Drama 3 Tier 2 Five Languages")
    writer.add_string("audiocpp.model_spec.family", "fish_drama3")
    writer.add_uint32("audiocpp.model_spec.version", 1)
    writer.add_string("audiocpp.model_spec.json", spec.read_text(encoding="utf-8"))
    writer.add_string("general.source.url", f"https://huggingface.co/{SOURCE}/tree/{REVISION}")
    sidecars = {"config.json": json.dumps(config).encode()}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "LICENSE", "NOTICE"):
        sidecars[name] = (source / name).read_bytes()
    writer.add_array("audiocpp.embedded_files.names", list(sidecars))
    writer.add_array("audiocpp.embedded_files.contents", [v.decode("utf-8") for v in sidecars.values()])
    names, ranks, dims = [], [], []

    def add(name, data, shape, dtype=None):
        writer.add_tensor(f"_audiocpp.{len(names)}", np.ascontiguousarray(data), raw_dtype=dtype)
        names.append(name)
        ranks.append(len(shape))
        dims.extend(shape)

    with ExitStack() as files:
        shards = {f: files.enter_context(safe_open(source / f, framework="pt", device="cpu"))
                  for f in sorted(set(weights.values()))}

        def get(key):
            return shards[weights[key]].get_tensor(key)

        for name in weights:
            if ".experts." in name or name.endswith("_scale_inv"):
                continue
            tensor = get(name)
            if name.endswith(".mlp.gate.weight"):
                layer = int(name.split("layers.")[1].split(".")[0])
                tensor = tensor[retained[layer]]
            if name + "_scale_inv" in weights:
                scale = get(name + "_scale_inv")
                rows, cols = tensor.shape
                if scale.shape != (rows, cols // 64):
                    raise ValueError(f"Unexpected FP8 scales for {name}")
                tensor = (tensor.float().reshape(rows, -1, 64) * scale[:, :, None]).reshape(rows, cols)
                data = q8(tensor.numpy())
                add("model_weights/" + name, data, tensor.shape, Q.Q8_0)
            elif name == "model.language_model.embed_tokens.weight" or (
                    not reference and name.startswith("audio_decoder") and tensor.ndim == 2
                    and tensor.shape[1] % 32 == 0 and tensor.shape[0] >= 128):
                data = np.empty((len(tensor), tensor.shape[1] // 32 * 34), np.uint8)
                for offset in range(0, len(tensor), 1024):
                    data[offset:offset + 1024] = q8(tensor[offset:offset + 1024].float().numpy())
                add("model_weights/" + name, data, tensor.shape, Q.Q8_0)
            else:
                accurate = tensor.ndim <= 1 or tensor.dtype == torch.float32 or name.endswith(
                    (".mlp.gate.weight", ".mlp.shared_expert_gate.weight"))
                add("model_weights/" + name, (tensor.float() if accurate else tensor.half()).numpy(), tensor.shape)

        for layer, ids in enumerate(retained):
            prefix = f"model.language_model.layers.{layer}.mlp.experts."
            for projection in ("gate_proj", "up_proj", "down_proj"):
                rows, cols = (2048, 512) if projection == "down_proj" else (512, 2048)
                qtype = Q.NVFP4 if reference else Q.Q3_K
                block_size, block_bytes = gguf.GGML_QUANT_SIZES[qtype]
                bank = np.empty((len(ids), rows, cols // block_size * block_bytes), np.uint8)
                globals_ = np.empty((len(ids), 1, 1), np.float32)
                for slot, original in enumerate(ids):
                    key = f"{prefix}{original}.{projection}.weight"
                    raw = get(key).numpy()
                    scales = get(key + "_scale").view(torch.uint8).numpy()
                    global_scale = get(key + "_scale_2").item()
                    bank[slot] = (repack_int3(raw, scales) if reference else compact_int3(raw, scales, global_scale))
                    globals_[slot, 0, 0] = global_scale
                name = f"model_weights/{prefix}{projection}.weight"
                add(name, bank, (len(ids), rows, cols), qtype)
                if reference:
                    add(name + ".gguf_scale", globals_, globals_.shape)
            print(f"Packed layer {layer + 1}/40 ({len(ids)} experts)", flush=True)

    codec = torch.load(source / "codec.pth", map_location="cpu", weights_only=True, mmap=True)
    for name, value in codec.items():
        if name.endswith((".parametrizations.weight.original0", ".weight_g")):
            continue
        if name.endswith((".parametrizations.weight.original1", ".weight_v")):
            value = value.float()
            legacy = name.endswith(".weight_v")
            scale = codec[name[:-1] + ("g" if legacy else "0")].float()
            value = value * (scale / torch.linalg.vector_norm(value, dim=tuple(range(1, value.ndim)), keepdim=True))
            name = name.removesuffix(".weight_v" if legacy else ".parametrizations.weight.original1") + ".weight"
        value = value.float() if value.ndim <= 1 else value.half()
        add("codec_weights/" + name, value.numpy(), value.shape)
    writer.add_array("audiocpp.tensor_names", names)
    writer.add_key_value("audiocpp.tensor_ranks", ranks, V.ARRAY, V.INT32)
    writer.add_key_value("audiocpp.tensor_shapes", dims, V.ARRAY, V.INT64)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RUNTIME / "five-lang-source")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download", action="store_true", help="Download the fixed source revision before conversion")
    parser.add_argument("--reference", action="store_true", help="Build a larger reference package for quality comparisons")
    args = parser.parse_args()
    if args.output is None:
        args.output = RUNTIME / ("Drama3-Tier2-5Lang-Reference.gguf" if args.reference else FILENAME)
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(SOURCE, revision=REVISION, local_dir=args.source,
                          allow_patterns=list(json.loads((ROOT / "resources/five-lang-source.json").read_text())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as work:
        tempfile.tempdir = work
        output = Path(work) / FILENAME
        build(args.source, output, RUNTIME / "audio.cpp/model_specs/fish_drama3.json", args.reference)
        with output.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = Path(work) / "package.sha256"
        receipt.write_text(digest + "\n", encoding="ascii")
        output.replace(args.output)
        receipt.replace(args.output.with_suffix(".gguf.sha256"))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
