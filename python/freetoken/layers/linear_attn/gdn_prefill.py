from __future__ import annotations

import torch


def chunk_gated_delta_rule_owned(
    qkv: list[torch.Tensor],
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`` with the same kernels in the
    same order, for inference only; returns ``(o, h)``.

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
    idx = initial_state_indices.to(torch.int32)
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
            k=k, w=w, u=u, g=g, initial_state=initial_state, initial_state_indices=idx,
            cu_seqlens=cu, chunk_indices=chunk_indices,
        )
        w = u = None
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu)
    return o.to(q.dtype), h


__all__ = ["chunk_gated_delta_rule_owned"]
