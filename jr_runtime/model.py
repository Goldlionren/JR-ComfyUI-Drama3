"""Adapter for Fish Audio's unmodified, block-FP8 Drama 3 teacher."""
import json
from pathlib import Path
import sys
from types import MethodType

import torch

from .setup_runtime import runtime_dir
from .prompt import verify_control_tokens

GIB = 1024**3
# 6.87 GiB bf16 dense weights, approximately 1.8 GiB fp32 codec including
# buffers, plus 4 GiB for the KV caches and transient routed expert batches.
COLD_VRAM_TARGET = 13 * GIB


def make_room_for_teacher(device):
    """Ask an active Comfy model manager for only the cold-load working set."""
    free, _ = torch.cuda.mem_get_info(device)
    manager = sys.modules.get("comfy.model_management")
    if free < COLD_VRAM_TARGET and manager is not None:
        manager.free_memory(COLD_VRAM_TARGET, device)
        # The Comfy estimate includes inactive torch allocator blocks. Return
        # them to CUDA before reading the physical expert-placement budget.
        torch.cuda.empty_cache()


def activate_runtime():
    root = runtime_dir()
    source = root / "fish-speech"
    if not (source / "fish_speech/models/text2semantic/fish_omni/model.py").is_file():
        raise RuntimeError(f"Missing official teacher runtime in {source}. Run jr_runtime/setup_runtime.py.")
    loaded = sys.modules.get("fish_speech")
    # Upstream distributes fish_speech as a PEP 420 namespace package.
    if loaded and (source / "fish_speech").resolve() not in {
            Path(path).resolve() for path in getattr(loaded, "__path__", ())}:
        raise RuntimeError("A different fish_speech has already been imported. Start the JR teacher in an isolated process.")
    for path in (source, root / "vendor"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return source


def host_forward(experts, x, ids, weights):
    """Stage only the routed rows of a host-resident layer, preserving FP8 bytes."""
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward

    host_ids = ids.detach().to("cpu")
    if ids.shape[0] == 1:
        selected = host_ids.flatten()
        remapped = torch.arange(ids.shape[1], device=x.device).reshape_as(ids)
    else:
        selected, inverse = torch.unique(host_ids, sorted=True, return_inverse=True)
        remapped = inverse.to(x.device)
    def transfer(tensor):
        # numpy's byte gather does not start a CPU torch thread pool for
        # each tiny routing decision. View scales with their native dtype.
        raw = tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
        compact = torch.from_numpy(raw.numpy()[selected.numpy()])
        return compact.to(x.device).view(tensor.dtype)
    return fp8_moe_forward(x, remapped, weights, transfer(experts.w13_weight),
                           transfer(experts.w13_scale_inv), transfer(experts.w2_weight),
                           transfer(experts.w2_scale_inv))


def load_teacher(directory, device, cancel=None):
    activate_runtime()
    from fish_speech.models.text2semantic.fish_omni.backbone import (
        CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY, FishOmniBackbone,
        load_backbone_weights, module_bytes,
    )
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_gemm_supported
    from fish_speech.models.text2semantic.fish_omni.model import (
        AUDIO_DECODER_PREFIXES, FishOmniModel, iter_checkpoint_tensors,
    )
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights
    from tokenizers import Tokenizer

    directory = Path(directory)
    config = FishOmniConfig.from_dict(json.loads((directory / "config.json").read_text("utf-8")))
    if not config.fp8_experts or not fp8_gemm_supported([device]):
        raise RuntimeError("The FP8 teacher requires official bsgemm-moe weights, CUDA SM89+ and Triton.")
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=torch.bfloat16)
        tower = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
    layers = backbone.model.layers
    experts_size = sum(module_bytes(layer.mlp.experts) for layer in layers)
    dense_size = module_bytes(backbone) + module_bytes(tower) - experts_size
    free, _ = torch.cuda.mem_get_info(device)
    # Headroom for the codec, growing KV caches and temporary FP8 buffers.
    room = free - dense_size - 4 * 1024**3
    if room < 0:
        raise RuntimeError(f"FP8 teacher needs about 13 GiB free for a cold load; after the codec loaded, "
                           f"{free / GIB:.1f} GiB remained on {device}, below the "
                           f"{dense_size / GIB + 4:.1f} GiB needed for dense weights and inference.")
    backbone.model.embed_tokens.to_empty(device=device)
    backbone.model.norm.to_empty(device=device)
    backbone.lm_head.to_empty(device=device)
    offloaded = 0
    for layer in layers:
        if cancel:
            cancel()
        experts = layer.mlp.experts
        size = module_bytes(experts)
        # Skip the experts when materializing dense siblings.
        del layer.mlp.experts
        layer.to_empty(device=device)
        layer.mlp.experts = experts
        if size <= room:
            experts.to_empty(device=device)
            room -= size
        else:
            experts.to_empty(device="cpu")
            experts.forward = MethodType(host_forward, experts)
            offloaded += 1
    def tensors(skip_prefixes):
        for index, item in enumerate(iter_checkpoint_tensors(directory, skip_prefixes=skip_prefixes)):
            if cancel and index % 128 == 0:
                cancel()
            yield item
    load_backbone_weights(backbone, tensors(AUDIO_DECODER_PREFIXES))
    tower.to_empty(device=device)
    load_audio_decoder_weights(tower, tensors((CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY)))
    tower.reference_sampling = "ras"
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    verify_control_tokens(tokenizer)
    model = FishOmniModel(config, backbone, tower, tokenizer).eval()
    return model, offloaded


def load_codec(directory, device):
    source = activate_runtime()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    codec = instantiate(OmegaConf.load(str(source / "fish_speech/configs/modded_dac_vq.yaml")))
    state = torch.load(Path(directory) / "codec.pth", map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state)
    if any(key.startswith("generator.") for key in state):
        state = {key.removeprefix("generator."): value for key, value in state.items() if key.startswith("generator.")}
    rebuilt = {f"{name}.{key}".lstrip(".") for name, mod in codec.named_modules()
               for key in mod._non_persistent_buffers_set}
    codec.load_state_dict({key: value for key, value in state.items() if key not in rebuilt}, strict=True)
    for module in codec.modules():
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
        elif hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    return codec.eval().to(device=device, dtype=torch.float32)
