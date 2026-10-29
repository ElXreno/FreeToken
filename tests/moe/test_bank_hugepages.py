"""Host banks ask for huge pages only while free 2 MiB blocks cover them."""

from __future__ import annotations

import mmap

import pytest
from freetoken.moe import host_banks as hb

MIB2 = 2 << 20

BUDDYINFO = (
    "Node 0, zone      DMA      0      0      0      0      0      0      0      0      1      1      2\n"
    "Node 0, zone   Normal    100     50     20     10      5      3      2      1      4      0      3\n"
)


@pytest.fixture
def fresh(monkeypatch):
    if not hasattr(mmap, "MADV_HUGEPAGE"):
        pytest.skip("no MADV_HUGEPAGE")
    monkeypatch.setattr(hb, "_huge_budget", None)
    monkeypatch.setattr(hb, "_huge_denied", False)
    monkeypatch.delenv("FREETOKEN_BANK_HUGEPAGES", raising=False)
    return monkeypatch


def test_buddyinfo_counts_orders_from_nine(tmp_path, monkeypatch):
    path = tmp_path / "buddyinfo"
    path.write_text(BUDDYINFO)
    real_open = open
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: real_open(path if p == "/proc/buddyinfo" else p, *a, **k))
    dma = 1 * MIB2 + 2 * 2 * MIB2
    normal = 0 * MIB2 + 3 * 2 * MIB2
    assert hb._free_huge_bytes() == dma + normal


def test_auto_grants_banks_until_the_free_blocks_run_out(fresh):
    fresh.setattr(hb, "_free_huge_bytes", lambda: int(10 * MIB2 * hb._HUGE_MARGIN))
    assert hb._hugepages_wanted(6 * MIB2)
    assert hb._hugepages_wanted(4 * MIB2)
    assert not hb._hugepages_wanted(MIB2)


def test_auto_reads_buddyinfo_once(fresh):
    calls = []
    fresh.setattr(hb, "_free_huge_bytes", lambda: calls.append(1) or 100 * MIB2)
    for _ in range(3):
        hb._hugepages_wanted(MIB2)
    assert len(calls) == 1


def test_explicit_modes_ignore_the_free_blocks(fresh):
    fresh.setattr(hb, "_free_huge_bytes", lambda: 0)
    fresh.setenv("FREETOKEN_BANK_HUGEPAGES", "1")
    assert hb._hugepages_wanted(MIB2)
    fresh.setenv("FREETOKEN_BANK_HUGEPAGES", "0")
    assert not hb._hugepages_wanted(MIB2)
