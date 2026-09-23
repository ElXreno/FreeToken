"""Tiered cache: intermediate prefill chunks commit their GDN boundary snapshot (--prefix-cache-chunk-snapshots).

Overlap scheduling creates chunk N+1 (copying chunk N's cache handle) before chunk N drains, so a
commit at N's drain must hand the new handle to N+1 or N+1 later releases the span N gave the tree.
"""

from __future__ import annotations

import tempfile

import torch

from freetoken.core import SamplingParams
from freetoken.kvcache.host_tier import HostTier
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig

CHUNK = 8
N_CHUNKS = 4
WIDTH = 64
MAX_RUNNING = 4
PAGES = 256


class _KVPool:
    def __init__(self):
        self._kv_buffer = torch.zeros((2, 2, PAGES, 1, 2, 8), dtype=torch.uint8)
        self.device = torch.device("cpu")
        self.dtype = torch.uint8


def _setup(directory):
    from freetoken.core import Context, get_global_ctx, set_global_ctx
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=16, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1)
    tier = HostTier(directory, 64 << 20, _KVPool(), pool, {"model": "fake"})
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32)
    cm = CacheManager(PAGES, 1, pt, "hybrid_radix", linear_state_pool=pool, host_tier=tier)
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=1))
    return cm, tm, pm, tier


def _drive(cm, tm, pm, prompt):
    """Chunked prefill with the real overlap order: schedule chunk N+1, then drain chunk N."""
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.scheduler.utils import PendingReq

    pm.pending_list = [PendingReq(uid=7, input_ids=prompt, sampling_params=SamplingParams(max_tokens=4))]
    last, final = None, None
    while pm.runnable or last is not None:
        batch = pm.schedule_next_batch(CHUNK)
        if batch is not None:
            for r in batch.reqs:
                r.mamba_last_track_seqlen = r.device_len
                r.mamba_next_track_idx ^= 1
            cm.allocate_paged(batch.reqs)
            for r in batch.reqs:
                r.complete_one()
        if last is not None:
            for r in last.reqs:
                cm.cache_req(r, finished=False)
                if not isinstance(r, ChunkedReq):
                    final = r
        last = batch
    cm.cache_req(final, finished=True)
    tm.free(final.table_idx)


def test_intermediate_chunk_commits_keep_page_accounting_whole():
    with tempfile.TemporaryDirectory() as d:
        cm, tm, pm, tier = _setup(d)
        prompt = torch.arange(100, 100 + N_CHUNKS * CHUNK, dtype=torch.int32)
        _drive(cm, tm, pm, prompt)
        cm.check_integrity()
        assert cm.prefix_cache.full_protected == 0
        assert cm.prefix_cache.host_stats()["snapshots"] >= N_CHUNKS - 1
        tier.close()


def test_a_request_sharing_two_chunks_resumes_from_the_second_boundary():
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as d:
        cm, tm, pm, tier = _setup(d)
        prompt = torch.arange(100, 100 + N_CHUNKS * CHUNK, dtype=torch.int32)
        _drive(cm, tm, pm, prompt)
        other = torch.cat([prompt[: 2 * CHUNK], torch.arange(900, 900 + 2 * CHUNK, dtype=torch.int32)])
        m = cm.match_req(SimpleNamespace(input_ids=other, input_len=len(other)))
        assert m.cuda_handle.cached_len == 2 * CHUNK
        tier.close()
