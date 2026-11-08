"""llama.cpp qwen35 / qwen35moe GGUFs with the MTP head the converter keeps as a block past the decoder (conversion/qwen.py ``_QwenMtpMixin``).

Tiny files written with gguf-py, laid out as the converter writes them: ``block_count`` counts the head, ``nextn_predict_layers`` says how many of the blocks it is, and the head's own tensors sit on its block as ``nextn.*``. The reader has to fill exactly the state dict of the model the engine builds with the draft head, the same one an HF checkpoint's ``mtp.*`` tensors fill.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights, parse_gguf_config, to_gguf

gguf = pytest.importorskip("gguf")
T = gguf.GGMLQuantizationType
quantize = gguf.quants.quantize

H, V = 64, 64
QH, KVH, AHD = 2, 1, 64
NK, NV, DK, DV = 1, 2, 16, 16
I, E, MI, SI = 64, 4, 32, 64
MTP = "blk.2"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _write(path: str, moe: bool) -> dict[str, np.ndarray]:
    """A two-layer decoder (GDN, attention) and the MTP head as block 2; returns the float values of every tensor."""
    arch = "qwen35moe" if moe else "qwen35"
    w = gguf.GGUFWriter(path, arch)
    for key, value in {
        "block_count": 3, "nextn_predict_layers": 1, "context_length": 4096, "embedding_length": H,
        "attention.head_count": QH, "attention.head_count_kv": KVH, "attention.key_length": AHD, "attention.value_length": AHD,
        "rope.dimension_count": 16, "ssm.conv_kernel": 4, "ssm.group_count": NK, "ssm.time_step_rank": NV,
        "ssm.state_size": DK, "ssm.inner_size": NV * DV,
        **({"expert_count": E, "expert_used_count": 2, "expert_feed_forward_length": MI, "expert_shared_feed_forward_length": SI} if moe else {"feed_forward_length": I}),
    }.items():
        w.add_uint32(f"{arch}.{key}", value)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32(f"{arch}.rope.freq_base", 1e7)
    w.add_array(f"{arch}.attention.recurrent_layers", [True, False, False])
    w.add_array(f"{arch}.rope.dimension_sections", [4, 2, 2, 0])
    rng = np.random.default_rng(0)
    values: dict[str, np.ndarray] = {}

    def add(name: str, *shape: int, quantized: bool = False) -> None:
        values[name] = rng.standard_normal(shape, dtype=np.float32)
        if quantized:
            w.add_tensor(name, quantize(values[name], T.Q8_0), raw_dtype=T.Q8_0)
        else:
            w.add_tensor(name, values[name])

    def block(blk: str, linear: bool) -> None:
        add(f"{blk}.attn_norm.weight", H)
        add(f"{blk}.post_attention_norm.weight", H)
        if linear:
            add(f"{blk}.attn_qkv.weight", 2 * NK * DK + NV * DV, H, quantized=True)
            add(f"{blk}.attn_gate.weight", NV * DV, H, quantized=True)
            add(f"{blk}.ssm_beta.weight", NV, H)
            add(f"{blk}.ssm_alpha.weight", NV, H)
            add(f"{blk}.ssm_conv1d.weight", 2 * NK * DK + NV * DV, 4)
            add(f"{blk}.ssm_dt.bias", NV)
            values[f"{blk}.ssm_a"] = -np.abs(rng.standard_normal(NV, dtype=np.float32)) - 0.1
            w.add_tensor(f"{blk}.ssm_a", values[f"{blk}.ssm_a"])
            add(f"{blk}.ssm_norm.weight", DV)
            add(f"{blk}.ssm_out.weight", H, NV * DV, quantized=True)
        else:
            add(f"{blk}.attn_q.weight", 2 * QH * AHD, H, quantized=True)
            add(f"{blk}.attn_k.weight", KVH * AHD, H, quantized=True)
            add(f"{blk}.attn_v.weight", KVH * AHD, H, quantized=True)
            add(f"{blk}.attn_output.weight", H, QH * AHD, quantized=True)
            add(f"{blk}.attn_q_norm.weight", AHD)
            add(f"{blk}.attn_k_norm.weight", AHD)
        if moe:
            add(f"{blk}.ffn_gate_inp.weight", E, H)
            for p, shape in (("gate", (E, MI, H)), ("up", (E, MI, H)), ("down", (E, H, MI))):
                add(f"{blk}.ffn_{p}_exps.weight", *shape, quantized=True)
            add(f"{blk}.ffn_gate_inp_shexp.weight", H)
            for p, shape in (("gate", (SI, H)), ("up", (SI, H)), ("down", (H, SI))):
                add(f"{blk}.ffn_{p}_shexp.weight", *shape, quantized=True)
        else:
            for p, shape in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
                add(f"{blk}.ffn_{p}.weight", *shape, quantized=True)

    add("token_embd.weight", V, H, quantized=True)
    add("output.weight", V, H, quantized=True)
    add("output_norm.weight", H)
    block("blk.0", True)
    block("blk.1", False)
    block(MTP, False)
    add(f"{MTP}.nextn.eh_proj.weight", H, 2 * H, quantized=True)
    for leaf in ("enorm", "hnorm", "shared_head_norm"):
        add(f"{MTP}.nextn.{leaf}.weight", H)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return values


@pytest.fixture(scope="module", params=["dense", "moe"])
def model(request, tmp_path_factory):
    path = str(tmp_path_factory.mktemp(request.param) / "model.gguf")
    return request.param == "moe", path, _write(path, request.param == "moe")


def _meta_state_dict(path: str, moe: bool, mtp: bool) -> dict[str, torch.Tensor]:
    """State dict of the model the engine builds for ``path`` (routed experts offloaded), the draft head with ``mtp``, on the meta device."""
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    strategy = "offload" if moe else "auto"
    config = EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy=strategy, mtp_draft=mtp,
                          mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)))
    object.__setattr__(config.model_config, "moe_strategy", strategy)
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    assert config.model_config.mtp_draft is mtp
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return create_model(config.model_config).state_dict()
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


def _load(path: str, mtp: bool) -> dict[str, torch.Tensor]:
    return dict(iter_gguf_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True, include_vision=False, include_mtp=mtp))


def test_the_nextn_block_is_the_mtp_head_not_a_decoder_layer(model):
    _moe, path, _values = model
    config = parse_gguf_config(build_gguf_shim(path))
    assert config.num_layers == 2 and len(config.attention_groups) == 2
    assert config.mtp_num_layers == 1 and not config.mtp_dedicated_embeddings


@pytest.mark.parametrize("mtp", [False, True])
def test_emitted_keys_are_the_model_state_dict(model, mtp):
    """Every buffer of the model, the draft head's when it is built, is filled with its shape; a quantized head module stays packed."""
    moe, path, _values = model
    loaded, state = _load(path, mtp), _meta_state_dict(path, moe, mtp)
    assert set(loaded) == set(state)
    for key, tensor in loaded.items():
        assert tensor.shape == state[key].shape, key
        assert (tensor.dtype is torch.uint8) is (state[key].dtype is torch.uint8), key
    assert any(k.startswith("mtp.") for k in loaded) is mtp
    if mtp:
        assert loaded["mtp.fc.weight"].dtype is torch.uint8
        assert loaded["mtp.layers.0.self_attn.qkv_proj.weight"].dtype is torch.uint8


def test_head_tensors_come_from_its_block_with_the_converters_norm_offset_undone(model):
    moe, path, values = model
    loaded = _load(path, True)
    index = {name: 0 for name in values}
    mlp = {
        "mtp.layers.0.mlp.shared_expert.down_proj": f"{MTP}.ffn_down_shexp",
        "mtp.layers.0.mlp.experts.0.gate_proj": f"{MTP}.ffn_gate_exps",
    } if moe else {"mtp.layers.0.mlp.down_proj": f"{MTP}.ffn_down"}
    for name, tensor in {
        "mtp.fc": f"{MTP}.nextn.eh_proj",
        "mtp.layers.0.self_attn.q_proj": f"{MTP}.attn_q",
        "mtp.layers.0.self_attn.o_proj": f"{MTP}.attn_output",
        **mlp,
        "model.language_model.layers.1.self_attn.q_proj": "blk.1.attn_q",
    }.items():
        assert to_gguf(name, index) == (tensor,), name
    for state, tensor in (
        ("mtp.pre_fc_norm_embedding.weight", "nextn.enorm"),
        ("mtp.pre_fc_norm_hidden.weight", "nextn.hnorm"),
        ("mtp.norm.weight", "nextn.shared_head_norm"),
        ("mtp.layers.0.input_layernorm.weight", "attn_norm"),
        ("mtp.layers.0.self_attn.k_norm.weight", "attn_k_norm"),
    ):
        assert torch.equal(loaded[state], torch.from_numpy(values[f"{MTP}.{tensor}.weight"]) - 1.0), state
    packed = quantize(values[f"{MTP}.nextn.eh_proj.weight"], T.Q8_0)
    assert torch.equal(loaded["mtp.fc.weight"][: packed.shape[0]], torch.from_numpy(packed))
    qkv = np.concatenate([quantize(values[f"{MTP}.attn_{p}.weight"], T.Q8_0) for p in "qkv"])
    assert torch.equal(loaded["mtp.layers.0.self_attn.qkv_proj.weight"][: qkv.shape[0]], torch.from_numpy(qkv))
