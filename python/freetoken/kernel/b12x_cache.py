"""On-disk memo for flashinfer's SM12x W4A16 CuTe-DSL compiles.

flashinfer keeps ``cute.compile`` results only in memory, so after every restart the first
prefill of each block-size bucket compiles again (0.7-2.4 s each on sm_120). A compiled
function dumps to a relocatable object that ``cutlass.runtime.load_module`` loads in
milliseconds; this swaps flashinfer's memo dict for one backed by those objects.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from typing import Any

from freetoken.env import ENV
from freetoken.utils import init_logger

logger = init_logger(__name__)

_INSTALLED = False


def _cache_root() -> str | None:
    setting = ENV.B12X_CACHE.value
    if setting == "off":
        return None
    if setting:
        return setting
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "freetoken", "b12x")


def _build_tag() -> str:
    """Objects are only valid for the exact flashinfer + CuTe DSL build and GPU arch that made them."""
    import cutlass
    import flashinfer
    import torch

    cc = torch.cuda.get_device_capability()
    ident = [
        flashinfer.__version__,
        os.path.dirname(os.path.realpath(flashinfer.__file__)),
        cutlass.__version__,
        os.path.dirname(os.path.realpath(cutlass.__file__)),
        f"sm_{cc[0]}{cc[1]}",
    ]
    return hashlib.sha256(json.dumps(ident).encode()).hexdigest()[:16]


class DiskBackedCompiles(dict):
    """flashinfer's ``_COMPILE_CACHE``: a miss in memory falls through to ``<dir>/<key hash>.o``, a new entry is written there."""

    def __init__(self, base: dict, directory: str) -> None:
        super().__init__(base)
        self.directory = directory
        self._modules: list[Any] = []
        self._write_failed = False

    def _entry(self, key: Any) -> tuple[str, str]:
        digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]
        return os.path.join(self.directory, f"{digest}.o"), f"ft_{digest}"

    def get(self, key: Any, default: Any = None) -> Any:
        hit = super().get(key)
        if hit is not None:
            return hit
        path, prefix = self._entry(key)
        if not os.path.exists(path):
            return default
        try:
            from cutlass import runtime

            module = runtime.load_module(path)
            fn = getattr(module, prefix)
        except Exception as exc:  # noqa: BLE001 -- a stale or torn object only costs a recompile
            logger.warning(f"b12x compile cache: dropping unloadable {path} ({exc!r})")
            with contextlib.suppress(OSError):
                os.unlink(path)
            return default
        self._modules.append(module)
        super().__setitem__(key, fn)
        return fn

    def __setitem__(self, key: Any, fn: Any) -> None:
        super().__setitem__(key, fn)
        path, prefix = self._entry(key)
        try:
            blob = fn.dump_to_object(prefix)
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 -- persisting is best effort
            if not self._write_failed:
                logger.warning(f"b12x compile cache: cannot persist to {self.directory} ({exc!r})")
                self._write_failed = True


def install() -> None:
    """Back flashinfer's b12x compile memo with the disk cache (idempotent; any failure leaves it in memory only)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    root = _cache_root()
    if root is None:
        return
    try:
        import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_compiler as compiler

        tag = _build_tag()
        directory = os.path.join(root, tag)
        os.makedirs(directory, exist_ok=True)
        for entry in os.listdir(root):
            if entry != tag:
                shutil.rmtree(os.path.join(root, entry), ignore_errors=True)
        with compiler._COMPILE_CACHE_LOCK:
            compiler._COMPILE_CACHE = DiskBackedCompiles(compiler._COMPILE_CACHE, directory)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"b12x compile cache disabled ({exc!r})")
        return
    logger.info(f"b12x compile cache: {directory}")
