"""A verify step holds a second GDN slot per request, and the pool has to be sized for it.

Without the extra slot the scheduler allocates one it does not have and the worker dies with
"LinearStatePool exhausted: need 1, have 0" mid-request -- and only once two requests run at
once, which is why a single-request benchmark never sees it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from freetoken.kvcache import linear_state_pool as pool_mod
from freetoken.kvcache.linear_state_pool import (
    LinearStatePool,
    _linear_pool_min_slots,
    _linear_pool_num_slots,
)


def cfg(**kw) -> SimpleNamespace:
    base = dict(
        max_running_req=2, cache_type="hybrid_radix", prefix_cache_dir=None,
        linear_state_cache_ratio=2.0, mtp_verify=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_host_tier_leaves_room_for_the_second_slot():
    """The shipped shape: host tier on, two requests. 3*mr+1 is what starved the scheduler."""
    plain = _linear_pool_num_slots(cfg(prefix_cache_dir="/var/lib/freetoken/arena"))
    spec = _linear_pool_num_slots(cfg(prefix_cache_dir="/var/lib/freetoken/arena", mtp_verify=True))
    assert plain == 7, plain
    assert spec == plain + 2, spec
    # usable = physical - padding sink; every request needs live + 2 ping-pong + verify
    assert spec - 1 >= 4 * 2


def test_every_branch_grows_by_one_per_request():
    for extra in ({}, {"prefix_cache_dir": "/arena"}, {"cache_type": "naive"}):
        for mr in (1, 2, 4):
            off = _linear_pool_num_slots(cfg(max_running_req=mr, **extra))
            on = _linear_pool_num_slots(cfg(max_running_req=mr, mtp_verify=True, **extra))
            assert on - off == mr, (extra, mr, off, on)


def test_floor_grows_too():
    """The rebuild floor must move with the pool, or a rebuild rejects a size verify needs."""
    for extra in ({}, {"prefix_cache_dir": "/arena"}, {"cache_type": "naive"}):
        off = _linear_pool_min_slots(cfg(**extra))
        on = _linear_pool_min_slots(cfg(mtp_verify=True, **extra))
        assert on - off == 2, (extra, off, on)
        assert _linear_pool_num_slots(cfg(mtp_verify=True, **extra)) >= on


def test_num_free_slots_is_read_as_a_property_everywhere():
    """It is a property, so `pool.num_free_slots()` raises TypeError only once the pool runs dry.

    That is the one path the guard exists for, so no ordinary run reaches it: the scheduler
    shipped, served fine, and died the first time two requests filled the pool.
    """
    assert isinstance(inspect.getattr_static(LinearStatePool, "num_free_slots"), property)
    root = Path(pool_mod.__file__).parents[2]
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "num_free_slots"
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"num_free_slots called as a method at {offenders}"


def test_verify_off_is_untouched():
    """Nobody who leaves --mtp-verify off pays a slot for it."""
    for extra in ({}, {"prefix_cache_dir": "/arena"}, {"cache_type": "naive"}):
        assert _linear_pool_num_slots(cfg(**extra)) == _linear_pool_num_slots(
            cfg(mtp_verify=False, **extra)
        )
