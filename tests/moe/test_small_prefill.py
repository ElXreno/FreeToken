"""Small prefill chunks take the hybrid decode path (--moe-hybrid-small-prefill)."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("limit", "rows", "target", "cpu_layer", "expected"),
    [
        (64, 40, "hybrid", False, True),
        (64, 64, "hybrid", False, True),
        (64, 65, "hybrid", False, False),
        (0, 1, "hybrid", False, False),
        (64, 40, "gpu", False, False),
        (64, 40, "hybrid", True, False),
    ],
)
def test_small_prefill_routes_through_the_hybrid_decode_path(limit, rows, target, cpu_layer, expected):
    from freetoken.layers.moe import OffloadMoELayer

    cache = SimpleNamespace(decode_target=target, small_prefill_rows=limit, is_cpu_layer=lambda layer_id: cpu_layer)
    layer = SimpleNamespace(offload_cache=cache, layer_id=3)
    assert OffloadMoELayer._small_prefill_as_decode(layer, rows) is expected


def test_small_prefill_needs_an_offload_cache():
    from freetoken.layers.moe import OffloadMoELayer

    assert OffloadMoELayer._small_prefill_as_decode(SimpleNamespace(offload_cache=None, layer_id=0), 8) is False
