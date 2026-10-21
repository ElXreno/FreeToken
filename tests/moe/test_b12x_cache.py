"""b12x compile objects on disk: a second process must run the fused MoE from them without
compiling and get the first process's output bit for bit; an unloadable object is dropped
and recompiled instead of failing the forward."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

HERE = Path(__file__).resolve().parent

SCRIPT = textwrap.dedent("""
    import os, sys, torch
    sys.path.insert(0, {here!r})
    import test_nvfp4_backends as T
    from freetoken.layers.quantization.moe.nvfp4 import _b12x_unusable_reason, b12x_fused_experts

    if _b12x_unusable_reason(torch.cuda.get_device_capability()) is not None:
        sys.exit(77)
    if os.environ["MODE"] == "forbid":
        import cutlass.cute as cute

        def forbidden(*a, **kw):
            raise RuntimeError("compiled although the objects are on disk")

        cute.compile = forbidden
    dev = torch.device("cuda")
    cache, _ = T._packed_cache(dev, "b12x", seed=8)
    torch.manual_seed(2)
    M = 300
    hidden = torch.randn(M, T.H, dtype=torch.bfloat16, device=dev) / 4
    w = torch.rand(M, T.TOPK, dtype=torch.float32, device=dev)
    ids = torch.randint(0, T.E, (M, T.TOPK), dtype=torch.int32, device=dev)
    cache.materialize_layer(0)
    cache.copy_missing()
    g1, g2 = cache.alphas_for_layer(0)
    b = cache.bank_views(T.E)
    out = b12x_fused_experts(hidden, b[0], b[1], g1, b[2], b[3], g2, w, ids, "silu", False)
    torch.save(out.cpu(), os.environ["OUT"])
""")


def _run(tmp_path: Path, mode: str, out: str) -> None:
    env = dict(os.environ, MODE=mode, OUT=str(tmp_path / out), FREETOKEN_B12X_CACHE=str(tmp_path / "b12x"))
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(here=str(HERE))],
        env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    if proc.returncode == 77:
        pytest.skip("b12x not runnable here")
    assert proc.returncode == 0, proc.stderr[-3000:]


def _objects(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "b12x").glob("*/*.o"))


def test_second_process_runs_from_disk_without_compiling(tmp_path):
    _run(tmp_path, "compile", "first.pt")
    assert _objects(tmp_path)
    _run(tmp_path, "forbid", "second.pt")
    assert torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "second.pt"))


def test_unloadable_object_is_recompiled(tmp_path):
    _run(tmp_path, "compile", "first.pt")
    objects = _objects(tmp_path)
    assert objects
    for obj in objects:
        obj.write_bytes(b"torn")
    _run(tmp_path, "compile", "second.pt")
    assert torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "second.pt"))
    assert all(obj.read_bytes() != b"torn" for obj in objects)
