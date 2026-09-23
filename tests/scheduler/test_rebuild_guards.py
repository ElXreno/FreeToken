"""A runtime cache rebuild is refused up front where it would leave live pointers into freed pools."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.engine.engine import CacheRebuildRejected, Engine
from freetoken.scheduler.scheduler import Scheduler


def test_engine_refuses_a_rebuild_while_mtp_draft_graphs_exist():
    eng = SimpleNamespace(_mtp_graphs={1: object()}, config=SimpleNamespace(), moe_offload_cache=None)
    with pytest.raises(CacheRebuildRejected, match="MTP draft head"):
        Engine.rebuild_runtime_cache(eng, moe_cache_size=1000)
    # without draft graphs the same request reaches the ordinary geometry checks
    eng._mtp_graphs = {}
    with pytest.raises(CacheRebuildRejected, match="no MoE offload cache"):
        Engine.rebuild_runtime_cache(eng, moe_cache_size=1000)


def _scheduler_stub(tiered: bool):
    calls = []
    return calls, SimpleNamespace(
        prefill_manager=SimpleNamespace(runnable=False),
        decode_manager=SimpleNamespace(runnable=False),
        cache_manager=SimpleNamespace(is_tiered=tiered, prefill_chunk_budget=None),
        device=torch.device("cuda"),
        config=SimpleNamespace(max_extend_tokens=4096, tp_info=SimpleNamespace(size=1)),
        engine=SimpleNamespace(rebuild_runtime_cache=lambda **kw: calls.append(kw)),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("pool", ["num_pages", "num_mamba_slots", "num_swa_pages"])
def test_scheduler_refuses_pool_resizes_under_the_host_tier(pool):
    calls, stub = _scheduler_stub(tiered=True)
    with pytest.raises(CacheRebuildRejected, match="host tier"):
        Scheduler.rebuild_cache(stub, **{pool: 64})
    assert calls == []
    # the expert cache is not mapped by the host tier, so resizing it alone still goes through
    Scheduler.rebuild_cache(stub, moe_cache_size=1000)
    assert calls and calls[0]["moe_cache_size"] == 1000
