"""Host tier under the hybrid radix cache: a file-backed arena that keeps every committed KV
span and GDN snapshot, plus the tree metadata that lets the cache survive a restart.

The arena is the source of truth and VRAM is a cache over it: commits write through, VRAM
eviction only drops the resident copy, a prefix hit promotes the missing spans back over
PCIe. Snapshots never occupy VRAM outside a running request's own slots; a hit restores
the matched snapshot straight into that request's live slot.
"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import mmap
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple

import torch

FORMAT_VERSION = 1
ALIGN = 4096
STAGE_ROWS = 1024
WRITEBACK_BYTES = 64 << 20
MAX_PENDING_BYTES = 1 << 30
SYNC_FILE_RANGE_WRITE = 2
_BITS_DTYPE = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def _load_sync_file_range():
    if os.environ.get("FREETOKEN_ARENA_WRITEBACK", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        fn = ctypes.CDLL(None, use_errno=True).sync_file_range
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint]
    fn.restype = ctypes.c_int
    return fn


_sync_file_range = _load_sync_file_range()
META_NAME = "meta.json"
ARENA_NAME = "arena.bin"


class HostRef(NamedTuple):
    offset: int
    nbytes: int
    # arena bytes this ref gives back on free; None = nbytes rounded up to ALIGN (whole allocations)
    reserved: int | None = None

    def released(self) -> int:
        return _round_up(max(self.nbytes, 1), ALIGN) if self.reserved is None else self.reserved

    def split(self, head: int) -> tuple[HostRef, HostRef]:
        """Two refs over [0, head) and [head, nbytes) that together release exactly this ref's extent."""
        total = self.released()
        return (
            HostRef(self.offset, head, head),
            HostRef(self.offset + head, self.nbytes - head, total - head),
        )


def _round_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class HostArena:
    """First-fit extent allocator over one sparse file mapped read-write."""

    def __init__(self, path: str, capacity: int, free_extents: list[tuple[int, int]] | None = None) -> None:
        self.path = path
        self.capacity = _round_up(capacity, ALIGN)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        if os.fstat(self.fd).st_size < self.capacity:
            os.ftruncate(self.fd, self.capacity)
        self.mm = mmap.mmap(self.fd, self.capacity, access=mmap.ACCESS_WRITE)
        if free_extents is None:
            self.free_extents: list[list[int]] = [[0, self.capacity]]
        else:
            self.free_extents = sorted([list(e) for e in free_extents if e[0] < self.capacity])
            for e in self.free_extents:
                e[1] = min(e[1], self.capacity - e[0])
        self.used = self.capacity - sum(size for _, size in self.free_extents)
        self._dirty_lo, self._dirty_hi, self._dirty_bytes = self.capacity, 0, 0
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="arena-writer")
        self._pending: list[tuple[int, int, int, Future]] = []

    def alloc(self, nbytes: int) -> HostRef | None:
        need = _round_up(max(nbytes, 1), ALIGN)
        for i, (off, size) in enumerate(self.free_extents):
            if size >= need:
                if size == need:
                    del self.free_extents[i]
                else:
                    self.free_extents[i] = [off + need, size - need]
                self.used += need
                return HostRef(off, nbytes, need)
        return None

    def free(self, ref: HostRef) -> None:
        off, size = ref.offset, ref.released()
        if size == 0:
            return
        self._wait(off, off + size)
        self.used -= size
        ext = self.free_extents
        lo, hi = 0, len(ext)
        while lo < hi:
            mid = (lo + hi) // 2
            if ext[mid][0] < off:
                lo = mid + 1
            else:
                hi = mid
        ext.insert(lo, [off, size])
        if lo + 1 < len(ext) and ext[lo][0] + ext[lo][1] == ext[lo + 1][0]:
            ext[lo][1] += ext[lo + 1][1]
            del ext[lo + 1]
        if lo > 0 and ext[lo - 1][0] + ext[lo - 1][1] == ext[lo][0]:
            ext[lo - 1][1] += ext[lo][1]
            del ext[lo]

    def view(self, ref: HostRef) -> torch.Tensor:
        return torch.frombuffer(self.mm, dtype=torch.uint8, count=ref.nbytes, offset=ref.offset)

    def write(self, ref: HostRef, src: torch.Tensor) -> None:
        """A device source lands asynchronously: the copy into pinned staging is queued on the
        current stream and the page-faulting copy into the file mapping runs on the writer thread,
        so the scheduler never waits on it. Reads, frees and flushes of the range wait for it."""
        src = src.contiguous()
        n = src.numel() * src.element_size()
        assert n <= ref.nbytes, f"host write {n} > extent {ref.nbytes}"
        dst = self.view(ref)[:n]
        if not src.is_cuda:
            dst.view(src.dtype).view(src.shape).copy_(src)
            return
        staging = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        staging.copy_(src.reshape(-1).view(torch.uint8), non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        self._submit(ref.offset, ref.offset + n, n, self._land, done, staging, dst)

    @staticmethod
    def _land(done: torch.cuda.Event, staging: torch.Tensor, dst: torch.Tensor) -> None:
        done.synchronize()
        dst.copy_(staging)

    def _submit(self, lo: int, hi: int, nbytes: int, fn: Callable[..., None], *args: Any) -> None:
        self._pending = [p for p in self._pending if not p[3].done()]
        while self._pending and sum(p[2] for p in self._pending) + nbytes > MAX_PENDING_BYTES:
            self._pending.pop(0)[3].result()
        self._pending.append((lo, hi, nbytes, self._writer.submit(fn, *args)))

    def _wait(self, lo: int = 0, hi: int | None = None) -> None:
        """Block until every queued write overlapping [lo, hi) has landed; re-raise its error."""
        hi = self.capacity if hi is None else hi
        for a, b, _, fut in self._pending:
            if a < hi and lo < b:
                fut.result()
        self._pending = [p for p in self._pending if not p[3].done()]

    def read(self, ref: HostRef, dst: torch.Tensor) -> None:
        n = dst.numel() * dst.element_size()
        assert n <= ref.nbytes, f"host read {n} > extent {ref.nbytes}"
        self._wait(ref.offset, ref.offset + n)
        dst.copy_(self.view(ref)[:n].view(dst.dtype).view(dst.shape))

    def writeback(self, offset: int, nbytes: int) -> None:
        """Start asynchronous writeback of a just-written range (no wait, no cache drop), queued
        behind the range's own landing copies on the writer thread.

        Without it every commit's pages stay dirty until memory pressure forces kswapd to
        write them back while the decode threads are running."""
        if _sync_file_range is None:
            return
        self._submit(offset, offset + nbytes, 0, self._writeback_now, offset, nbytes)

    def _writeback_now(self, offset: int, nbytes: int) -> None:
        self._dirty_lo = min(self._dirty_lo, offset)
        self._dirty_hi = max(self._dirty_hi, offset + nbytes)
        self._dirty_bytes += nbytes
        if self._dirty_bytes < WRITEBACK_BYTES:
            return
        lo = self._dirty_lo - self._dirty_lo % ALIGN
        _sync_file_range(self.fd, ctypes.c_int64(lo),
                         ctypes.c_int64(self._dirty_hi - lo), SYNC_FILE_RANGE_WRITE)
        self._dirty_lo, self._dirty_hi, self._dirty_bytes = self.capacity, 0, 0

    def flush(self) -> None:
        self._wait()
        self.mm.flush()
        os.fsync(self.fd)
        self._dirty_lo, self._dirty_hi, self._dirty_bytes = self.capacity, 0, 0

    def close(self) -> None:
        self._wait()
        self._writer.shutdown(wait=True)
        self.mm.close()
        os.close(self.fd)


class SnapSegment(NamedTuple):
    name: str
    tensor: torch.Tensor
    offset: int
    nbytes: int


class HostTier:
    """Arena-backed store for KV rows of the paged pool and whole GDN state slots."""

    def __init__(
        self,
        directory: str,
        capacity: int,
        kv_pool,
        linear_pool,
        validity_key: dict[str, Any],
        evict: Callable[[int], bool] | None = None,
    ) -> None:
        os.makedirs(directory, mode=0o750, exist_ok=True)
        self.directory = directory
        self.meta_path = os.path.join(directory, META_NAME)
        self.arena_path = os.path.join(directory, ARENA_NAME)
        self.key = validity_key
        self.kv_pool = kv_pool
        self.linear_pool = linear_pool
        self.evict = evict
        self.device = kv_pool.device
        buf = kv_pool._kv_buffer
        two, layers, pages, page_size, heads, head_dim = buf.shape
        bits = _BITS_DTYPE[buf.element_size()]
        self.kv_flat = buf.view(two, layers, pages * page_size, heads, head_dim).view(bits)
        self.kv_row_shape = (two, layers, heads, head_dim)
        self.row_bytes = two * layers * heads * head_dim * buf.element_size()
        self.stage_rows = STAGE_ROWS
        self._gather = torch.empty((two, layers, STAGE_ROWS, heads, head_dim), dtype=bits, device=self.device)
        self._rows = torch.empty((STAGE_ROWS, two, layers, heads, head_dim), dtype=bits, device=self.device)
        self.snap_segments: list[SnapSegment] = []
        off = 0
        if linear_pool is not None:
            for name, t in [("conv", linear_pool.conv_states), ("rec", linear_pool.recurrent_states),
                            *sorted(linear_pool.slot_states.items())]:
                nbytes = t[:, 0].numel() * t.element_size()
                self.snap_segments.append(SnapSegment(name, t, off, nbytes))
                off += _round_up(nbytes, 64)
        self.snap_bytes = off
        self.dirty = False
        self.last_flush = 0.0
        self.write_failures = 0
        loaded = self._load_meta_file()
        free = loaded["free"] if loaded is not None else None
        self.arena = HostArena(self.arena_path, capacity, free)
        self._loaded = loaded

    # ---------------------------------------------------------------- allocation
    def _alloc(self, nbytes: int) -> HostRef | None:
        ref = self.arena.alloc(nbytes)
        while ref is None and self.evict is not None and self.evict(nbytes):
            ref = self.arena.alloc(nbytes)
        if ref is None:
            self.write_failures += 1
        return ref

    def free(self, ref: HostRef | None) -> None:
        if ref is not None:
            self.arena.free(ref)
            self.dirty = True

    # ---------------------------------------------------------------- KV rows
    def put_kv(self, token_slots: torch.Tensor) -> HostRef | None:
        n = int(token_slots.numel())
        ref = self._alloc(n * self.row_bytes)
        if ref is None:
            return None
        idx = token_slots.long()
        for start, m, piece in self._chunks(ref, n):
            sel = idx[start : start + m]
            if m < self.stage_rows:
                sel = torch.cat([sel, sel[-1:].expand(self.stage_rows - m)])
            torch.index_select(self.kv_flat, 2, sel, out=self._gather)
            rows = self._rows[:m]
            rows.copy_(self._gather[:, :, :m].permute(2, 0, 1, 3, 4))
            self.arena.write(piece, rows)
        self.arena.writeback(ref.offset, ref.nbytes)
        self.dirty = True
        return ref

    def get_kv(self, ref: HostRef, token_slots: torch.Tensor) -> None:
        n = int(token_slots.numel())
        assert ref.nbytes == n * self.row_bytes, f"host span {ref.nbytes} != {n} rows"
        idx = token_slots.long()
        for start, m, piece in self._chunks(ref, n):
            rows = self._rows[:m]
            self.arena.read(piece, rows)
            gathered = self._gather[:, :, :m]
            gathered.copy_(rows.permute(1, 2, 0, 3, 4))
            self.kv_flat.index_copy_(2, idx[start : start + m], gathered)

    def _chunks(self, ref: HostRef, n: int) -> Iterator[tuple[int, int, HostRef]]:
        for start in range(0, n, self.stage_rows):
            m = min(self.stage_rows, n - start)
            yield start, m, HostRef(ref.offset + start * self.row_bytes, m * self.row_bytes)

    def split_kv(self, ref: HostRef, pos: int) -> tuple[HostRef, HostRef]:
        return ref.split(pos * self.row_bytes)

    # ---------------------------------------------------------------- GDN snapshots
    def put_snap(self, slot: int) -> HostRef | None:
        ref = self._alloc(self.snap_bytes)
        if ref is None:
            return None
        for seg in self.snap_segments:
            self.arena.write(HostRef(ref.offset + seg.offset, seg.nbytes), seg.tensor[:, slot])
        self.arena.writeback(ref.offset, ref.nbytes)
        self.dirty = True
        return ref

    def get_snap(self, ref: HostRef, slot: int) -> None:
        for seg in self.snap_segments:
            self.arena.read(HostRef(ref.offset + seg.offset, seg.nbytes), seg.tensor[:, slot])

    # ---------------------------------------------------------------- metadata
    def _load_meta_file(self) -> dict[str, Any] | None:
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return None
        if meta.get("format") != FORMAT_VERSION or meta.get("key") != self.key:
            return None
        if meta.get("row_bytes") != self.row_bytes or meta.get("snap_bytes") != self.snap_bytes:
            return None
        return meta

    def take_loaded_nodes(self) -> list[dict[str, Any]]:
        """Nodes recorded by the last flush whose key matches this server, once; [] otherwise."""
        loaded, self._loaded = self._loaded, None
        return list(loaded["nodes"]) if loaded is not None else []

    def read_nodes(self) -> list[dict[str, Any]]:
        meta = self._load_meta_file()
        return list(meta["nodes"]) if meta is not None else []

    def save_meta(self, nodes: list[dict[str, Any]]) -> None:
        self.arena.flush()
        meta = {
            "format": FORMAT_VERSION,
            "key": self.key,
            "saved_at": time.time(),
            "capacity": self.arena.capacity,
            "row_bytes": self.row_bytes,
            "snap_bytes": self.snap_bytes,
            "free": self.arena.free_extents,
            "nodes": nodes,
        }
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.meta_path)
        self.dirty = False
        self.last_flush = time.time()

    def stats(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "capacity_bytes": self.arena.capacity,
            "used_bytes": self.arena.used,
            "row_bytes": self.row_bytes,
            "snap_bytes": self.snap_bytes,
            "dirty": self.dirty,
            "last_flush": self.last_flush,
            "write_failures": self.write_failures,
        }

    def close(self) -> None:
        self.arena.close()


def encode_tokens(tokens: torch.Tensor) -> str:
    return base64.b64encode(tokens.to("cpu", torch.int32).contiguous().numpy().tobytes()).decode("ascii")


def decode_tokens(text: str) -> torch.Tensor:
    """Tree keys stay on the CPU: fast_compare_key walks them there."""
    raw = base64.b64decode(text.encode("ascii"))
    return torch.frombuffer(bytearray(raw), dtype=torch.int32).clone()


def _file_stamp(path: str) -> list[int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_size, int(st.st_mtime)]


def build_validity_key(config, kv_pool, linear_pool) -> dict[str, Any]:
    """Everything that changes the bytes a cached prefix would produce; a mismatch ignores the
    stored tree instead of feeding another model's state into this one."""
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.version import __version__

    model_path = os.path.realpath(config.model_path)
    key: dict[str, Any] = {
        "freetoken": __version__,
        "model_path": model_path,
        "config_json": _file_stamp(os.path.join(model_path, "config.json")),
        "weights_index": _file_stamp(os.path.join(model_path, "model.safetensors.index.json")),
        "page_size": config.page_size,
        "chunk_size": CHUNK_SIZE,
        "kv_dtype": str(kv_pool.dtype),
        "kv_shape": list(kv_pool._kv_buffer.shape[:2]) + list(kv_pool._kv_buffer.shape[4:]),
        "kv_scales": None,
    }
    scales = getattr(config, "kv_cache_scales", None)
    if scales:
        with open(scales, "rb") as f:
            key["kv_scales"] = hashlib.sha256(f.read()).hexdigest()
    direction = getattr(config, "ablate_direction", None)
    if direction:
        with open(direction, "rb") as f:
            key["ablate"] = [hashlib.sha256(f.read()).hexdigest(), config.ablate_layer, config.ablate_alpha]
    if linear_pool is not None:
        key["conv"] = [list(linear_pool.conv_states.shape[2:]), str(linear_pool.conv_states.dtype),
                       int(linear_pool.conv_states.shape[0])]
        key["rec"] = [list(linear_pool.recurrent_states.shape[2:]), str(linear_pool.recurrent_states.dtype),
                      int(linear_pool.recurrent_states.shape[0])]
        key["slot_states"] = sorted(
            [name, list(t.shape[2:]), str(t.dtype), int(t.shape[0])] for name, t in linear_pool.slot_states.items()
        )
    return key


__all__ = [
    "ALIGN",
    "FORMAT_VERSION",
    "HostArena",
    "HostRef",
    "HostTier",
    "build_validity_key",
    "decode_tokens",
    "encode_tokens",
]
