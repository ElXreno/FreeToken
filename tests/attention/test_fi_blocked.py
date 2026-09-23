"""Blocked bf16 prefill over an fp8 KV cache against flashinfer's own fp8 prefill."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HQ, HKV, D = 16, 2, 256
FP8 = torch.float8_e4m3fn


def _reference(q, k_cache, v_cache, cu_q, cu_k, indices):
    import flashinfer

    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 << 20, dtype=torch.uint8, device="cuda"), kv_layout="NHD", backend="fa2"
    )
    n = len(cu_q) - 1
    w.plan(
        qo_indptr=torch.tensor(cu_q, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(cu_k, dtype=torch.int32),
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.ones(n, dtype=torch.int32),
        num_qo_heads=HQ, num_kv_heads=HKV, head_dim_qk=D, page_size=1,
        q_data_type=torch.bfloat16, kv_data_type=FP8, causal=True,
    )
    return w.run(q, (k_cache, v_cache))


@pytest.fixture(scope="module")
def blocked():
    from freetoken.attention.fi_blocked import BlockedFp8Prefill

    return BlockedFp8Prefill(
        torch.empty(256 << 20, dtype=torch.uint8, device="cuda"),
        torch.empty(8 << 20, dtype=torch.uint8, device="cuda"),
        HQ, HKV, D, torch.bfloat16,
    )


@pytest.mark.parametrize(
    "reqs",
    [
        [(5000, 2048)],  # the past fits in the tail block
        [(16384, 1000)],  # exactly one full block, empty remainder
        [(2 * 16384 + 3000, 2048)],  # two full blocks and a remainder
        [(20000, 700), (0, 1348)],  # two requests in one chunk, one without a prefix
    ],
    ids=["tail-only", "one-block", "two-blocks-rem", "two-requests"],
)
def test_blocked_matches_the_fp8_kernel(blocked, reqs):
    from freetoken.attention.fi_blocked import BlockedReq

    g = torch.Generator(device="cuda").manual_seed(3)
    kv_lens = [p + q for p, q in reqs]
    slots = sum(kv_lens) + 4096
    k_cache = torch.randn(slots, 1, HKV, D, device="cuda", generator=g).to(FP8)
    v_cache = torch.randn(slots, 1, HKV, D, device="cuda", generator=g).to(FP8)
    # a real cache hands out slots wherever they are free
    indices = torch.randperm(slots, device="cuda", generator=g)[: sum(kv_lens)].to(torch.int32)
    q = torch.randn(sum(n for _, n in reqs), HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)

    cu_q, cu_k = [0], [0]
    for p, n in reqs:
        cu_q.append(cu_q[-1] + n)
        cu_k.append(cu_k[-1] + p + n)
    want = _reference(q, k_cache, v_cache, cu_q, cu_k, indices).float()

    plan = [
        BlockedReq(q0=cu_q[i], q1=cu_q[i + 1], k0=cu_k[i], prefix=p)
        for i, (p, _) in enumerate(reqs)
    ]
    blocked.plan(plan)
    got = blocked.run(q, k_cache, v_cache, indices, plan).float()

    rel = ((got - want).norm() / want.norm()).item()
    # both outputs are bf16; two roundings with a different summation order land near 4e-3
    assert rel < 1e-2, rel
    assert torch.isfinite(got).all()


def _tight_head() -> int:
    """The engine's head: flashinfer's split-KV bound for this geometry on this GPU + 32 MiB."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    bound = HQ * -(-2 * sms // HKV) * 64 * D * 4
    return -(-(bound + (32 << 20)) // (1 << 20)) << 20


@pytest.fixture(scope="module")
def blocked4096():
    from freetoken.attention.fi_blocked import BlockedFp8Prefill, scratch_bytes

    head = _tight_head()
    ws = torch.empty(head + scratch_bytes(4096, HQ, HKV, D, torch.bfloat16), dtype=torch.uint8, device="cuda")
    return BlockedFp8Prefill(ws, torch.empty(8 << 20, dtype=torch.uint8, device="cuda"),
                             HQ, HKV, D, torch.bfloat16, qcap=4096, head_bytes=head)


@pytest.mark.parametrize(
    "reqs",
    [[(20000, 4096)], [(2 * 16384 + 3000, 4096)], [(9000, 2048), (12000, 2048)]],
    ids=["tail-4096", "two-blocks-4096", "two-requests-2048"],
)
def test_a_4096_chunk_runs_blocked_in_a_workspace_sized_to_it(blocked4096, reqs):
    from freetoken.attention.fi_blocked import scratch_bytes

    assert blocked4096.qcap == 4096
    # the whole float workspace, head included, stays under the old flat 256 MiB
    assert _tight_head() + scratch_bytes(4096, HQ, HKV, D, torch.bfloat16) < 256 << 20
    test_blocked_matches_the_fp8_kernel(blocked4096, reqs)
