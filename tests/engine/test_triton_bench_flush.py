"""Triton's autotune L2-flush scratch fits the memory the KV pool leaves free."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_autotune_flush_scratch_is_two_l2():
    import triton
    from freetoken.engine.engine import _shrink_triton_bench_flush

    dev = torch.device("cuda", torch.cuda.current_device())
    _shrink_triton_bench_flush(dev)
    buf = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    l2 = torch.cuda.get_device_properties(dev).L2_cache_size
    assert buf.numel() * buf.element_size() == 2 * l2
    assert 2 * l2 < 256 * 1024 * 1024
