"""torch.profiler capture of whole forward steps, enabled by FREETOKEN_KPROF="decode=200".

One mode per process (torch allows a single active profiler). FREETOKEN_KPROF_SKIP forwards
of that mode are skipped first, then the next N are captured and written to
FREETOKEN_KPROF_OUT/kprof-<mode>.json as key_averages rows.
"""

from __future__ import annotations

import json
import os

import torch

from freetoken.env import ENV

_state: dict = {}


def _budget() -> dict[str, int]:
    spec = ENV.KPROF.value or ""
    return {k: int(v) for k, v in (p.split("=") for p in spec.split(",") if p)}


def _mode(batch) -> str:
    return "prefill" if batch.is_prefill else "decode"


def begin(batch) -> None:
    budget = _state.setdefault("budget", _budget())
    if not budget:
        return
    m = _mode(batch)
    if m not in budget or m in _state.setdefault("done", set()) or _state.get("active"):
        return
    seen = _state.setdefault("seen", {})
    seen[m] = seen.get(m, 0) + 1
    if seen[m] <= ENV.KPROF_SKIP.value:
        return
    p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    p.start()
    _state["active"] = (m, p, 0)


def end(batch) -> None:
    active = _state.get("active")
    if not active or active[0] != _mode(batch):
        return
    m, p, n = active
    n += 1
    if n < _state["budget"][m]:
        _state["active"] = (m, p, n)
        return
    torch.cuda.synchronize()
    p.stop()
    rows = [
        {
            "name": e.key,
            "device_us": float(getattr(e, "self_device_time_total", 0.0)),
            "cpu_us": float(e.self_cpu_time_total),
            "count": int(e.count),
        }
        for e in p.key_averages()
    ]
    out = ENV.KPROF_OUT.value or "."
    with open(os.path.join(out, f"kprof-{m}.json"), "w") as f:
        json.dump({"steps": n, "rows": rows}, f)
    _state["active"] = None
    _state["done"].add(m)
