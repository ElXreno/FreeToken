from __future__ import annotations

import torch


def gdn_prefill_chunk_fla(
    q: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, total, num_v_heads, head_v_dim] bf16
    g: torch.Tensor,        # [1, total, num_v_heads] log-decay (<=0), fp32
    beta: torch.Tensor,     # [1, total, num_v_heads] fp32
    *,
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,       # [num_seqs] slot id per sequence
    cu_seqlens: torch.Tensor,    # [num_seqs+1] int64
    scale: float,
    return_h: bool = False,
) -> torch.Tensor:
    """Chunked gated-delta-rule prefill via the vendored fla kernel. GQA is handled
    in-kernel (q/k at num_k_heads), q/k l2norm is done in-kernel, and the per-sequence
    recurrent state is read from and written back to ``state_source[indices]`` IN PLACE
    (no external l2norm, no Python stack of initial states, no copy_ writeback loop).
    Fresh sequences must have their ``state_source`` slot pre-zeroed by the caller.
    Returns ``o`` of shape ``[total, num_v_heads, head_v_dim]`` (bf16).

    When ``return_h=True`` also returns the per-chunk hidden-state buffer ``h`` of shape
    ``[1, NT_total, num_v_heads, head_v_dim, head_k_dim]`` (bf16). ``h[0, boh_i + c]`` is the
    recurrent state after ``c*64`` tokens of packed sequence ``i`` (chunk granularity 64), where
    ``boh_i = prepare_chunk_offsets(cu_seqlens, 64)[i]``. Note the last two dims are ``[V, K]`` --
    transposed vs ``state_source``'s ``[K, V]``. Used by the hybrid-radix track-checkpoint path."""
    from freetoken.kernel.fla import chunk_gated_delta_rule

    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=state_source, initial_state_indices=indices.to(torch.int32),
        cu_seqlens=cu_seqlens.to(torch.int64), head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    if return_h:
        return o[0], h  # h: [1, NT_total, num_v_heads, head_v_dim, head_k_dim]
    return o[0]  # [total, num_v_heads, head_v_dim]


def gdn_prefill_chunk_fla_owned(
    qkv: list[torch.Tensor],
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    state_source: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    return_h: bool = False,
):
    """``gdn_prefill_chunk_fla`` with the same kernels in the same order, for inference only.

    It takes q/k/v out of ``qkv`` (the caller keeps no other reference) and lets each tensor go
    once its last reader has run: raw q/k after the l2norm, v after the intra-chunk pass, w/u
    after the state pass. A long prefill chunk then peaks at about half the memory.
    """
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kernel.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from freetoken.kernel.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from freetoken.kernel.fla.chunk_o import chunk_fwd_o
    from freetoken.kernel.fla.cumsum import chunk_local_cumsum
    from freetoken.kernel.fla.index import prepare_chunk_indices
    from freetoken.kernel.fla.l2norm import l2norm_fwd
    from freetoken.kernel.fla.utils import custom_device_ctx

    q, k, v = (t.contiguous() for t in qkv)
    qkv.clear()
    assert q.dtype == k.dtype == v.dtype and q.dtype != torch.float32 and q.shape[0] == 1
    g, beta = g.contiguous(), beta.contiguous()
    cu = cu_seqlens.to(torch.int64)
    idx = indices.to(torch.int32)
    with custom_device_ctx(q.device.index):
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
        chunk_indices = prepare_chunk_indices(cu, CHUNK_SIZE)
        g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu, chunk_indices=chunk_indices)
        w, u, _ = chunk_gated_delta_rule_fwd_intra(
            k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_indices=chunk_indices,
        )
        v = None
        h, v_new = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g, initial_state=state_source, initial_state_indices=idx,
            cu_seqlens=cu, chunk_indices=chunk_indices,
        )
        w = u = None
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu)
    o = o.to(q.dtype)
    if return_h:
        return o[0], h
    return o[0]


def gdn_decode_fla(
    q: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, B, num_v_heads, head_v_dim] bf16
    a: torch.Tensor,        # [B, num_v_heads] raw
    b: torch.Tensor,        # [B, num_v_heads] raw
    *,
    A_log: torch.Tensor,        # [num_v_heads]
    dt_bias: torch.Tensor,      # [num_v_heads]
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,      # [B] int32 slot id per request
    cu_seqlens: torch.Tensor,   # [B+1] query indptr (arange) from FLAMetadata
    scale: float,
    out_indices: torch.Tensor | None = None,  # [B] int32 slots to land the new state in
) -> torch.Tensor:
    """Fused sigmoid-gating gated-delta-rule decode (vendored fla triton kernel): gating +
    in-kernel l2norm + recurrent update + state read/write-by-index in one kernel, with no
    external gating or gather/scatter/clone glue. Returns [B, num_v, V]."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,  # already fp32 (stored fp32)
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,  # already int32 (built int32 in the scheduler)
        initial_state_out_indices=out_indices,
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )
    # kernel returns o = [NK, *v.shape] then squeeze(NK) -> [1, B, num_v, V].
    # o[0] -> [B, num_v, V] (all B decode tokens; o[0,0] would drop B>1).
    return o[0]


__all__ = ["gdn_decode_fla", "gdn_prefill_chunk_fla", "gdn_prefill_chunk_fla_owned"]
