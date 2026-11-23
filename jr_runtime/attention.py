"""Select Gated DeltaNet kernels on the resident Drama 3 teacher."""
from functools import lru_cache
from importlib import import_module
from types import FunctionType, MethodType

from .setup_runtime import FLA_PACKAGE, runtime_dir


@lru_cache(maxsize=1)
def fla_rules():
    try:
        kernels = import_module(f"{FLA_PACKAGE}.ops.gated_delta_rule")
    except ModuleNotFoundError as exc:
        if exc.name == FLA_PACKAGE or (exc.name or "").startswith(FLA_PACKAGE + "."):
            package = runtime_dir() / "vendor" / FLA_PACKAGE
            try:
                with (package / "ops/gated_delta_rule/__init__.py").open("rb"):
                    pass
            except PermissionError as access:
                raise RuntimeError(f"启动 ComfyUI 的账户没有读取 FLA 内核的权限：{package}") from access
            except FileNotFoundError as missing:
                raise RuntimeError(f"找不到 FLA 内核：{package}。请使用 ComfyUI 的 Python 运行 "
                                   "jr_runtime/setup_runtime.py --fla。") from missing
        raise

    def adapt(kernel):
        def rule(q, k, v, decay, beta, initial_state=None):
            return kernel(q, k, v, g=decay, beta=beta, initial_state=initial_state,
                          output_final_state=True, use_qk_l2norm_in_kernel=True)
        return rule

    return adapt(kernels.chunk_gated_delta_rule), adapt(kernels.fused_recurrent_gated_delta_rule)


@lru_cache(maxsize=1)
def _fla_forward():
    from fish_speech.models.text2semantic.fish_omni.layers import GatedDeltaNet

    # Reuse the pinned upstream forward verbatim with two private kernel bindings.
    # Other model instances and the original module keep their own functions.
    forward = GatedDeltaNet.forward
    chunk, recurrent = fla_rules()
    bindings = dict(forward.__globals__, chunk_gated_delta_rule=chunk, recurrent_gated_delta_rule=recurrent)
    return FunctionType(forward.__code__, bindings, forward.__name__, forward.__defaults__, forward.__closure__)


def select_attention(model, use_fla):
    """Called under the synthesis lock, before a new prompt initializes its caches."""
    if getattr(model, "_jr_use_fla", False) == use_fla:
        return
    from fish_speech.models.text2semantic.fish_omni.layers import GatedDeltaNet

    blocks = [module for module in model.backbone.modules() if isinstance(module, GatedDeltaNet)]
    if not blocks:
        raise RuntimeError("当前教师没有可切换的 Gated DeltaNet 层。")
    if use_fla:
        forward = _fla_forward()
        for block in blocks:
            block.forward = MethodType(forward, block)
    else:
        for block in blocks:
            del block.forward
    model._jr_use_fla = use_fla
