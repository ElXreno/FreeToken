"""Device memory assembled from physical chunks with the CUDA virtual memory API.

A chunk (``cuMemCreate``) can be mapped into several reserved address ranges at once, so two
tensors may share bytes chunk by chunk: the MoE slot cache borrows the idle tail of the KV pool
this way (moe/kv_lend.py). Tensors returned here keep their address range alive.
"""

from __future__ import annotations

from contextlib import suppress
from functools import cache

import torch


def _driver():
    from cuda.bindings import driver

    return driver


def _call(result):
    cu = _driver()
    err, *rest = result
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err.name}")
    return rest[0] if len(rest) == 1 else None


def _index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _prop(device: torch.device):
    cu = _driver()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = _index(device)
    return prop


@cache
def _granularity(index: int) -> int:
    cu = _driver()
    torch.cuda.init()
    flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    return int(_call(cu.cuMemGetAllocationGranularity(_prop(torch.device("cuda", index)), flag)))


def granularity(device: torch.device) -> int:
    return _granularity(_index(device))


class Chunk:
    """One physical allocation of ``nbytes`` device memory, a multiple of the granularity."""

    def __init__(self, device: torch.device, nbytes: int) -> None:
        assert nbytes % granularity(device) == 0, nbytes
        self.nbytes = nbytes
        self.handle = _call(_driver().cuMemCreate(nbytes, _prop(device), 0))

    def __del__(self) -> None:
        with suppress(Exception):
            _driver().cuMemRelease(self.handle)


class Region:
    """A reserved address range; chunks are mapped into it, then ``tensor`` exposes the bytes."""

    def __init__(self, device: torch.device, nbytes: int) -> None:
        self.device = torch.device("cuda", _index(device))
        self.granule = granularity(device)
        self.nbytes = -(-nbytes // self.granule) * self.granule
        self.ptr = _call(_driver().cuMemAddressReserve(self.nbytes, self.granule, 0, 0))
        self.maps: list[tuple[int, Chunk]] = []

    def map(self, offset: int, chunk: Chunk) -> None:
        assert offset % self.granule == 0 and offset + chunk.nbytes <= self.nbytes, (offset, chunk.nbytes, self.nbytes)
        _call(_driver().cuMemMap(int(self.ptr) + offset, chunk.nbytes, 0, chunk.handle, 0))
        self.maps.append((offset, chunk))

    def tensor(self, nbytes: int) -> torch.Tensor:
        """``nbytes`` uint8 from the start of the range; bytes no chunk is mapped behind must never be touched."""
        cu = _driver()
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device.index
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        spans: list[list[int]] = []
        for offset, chunk in sorted(self.maps, key=lambda m: m[0]):
            if spans and spans[-1][1] == offset:
                spans[-1][1] += chunk.nbytes
            else:
                spans.append([offset, offset + chunk.nbytes])
        for lo, hi in spans:
            _call(cu.cuMemSetAccess(int(self.ptr) + lo, hi - lo, [access], 1))
        return torch.as_tensor(_ArrayInterface(int(self.ptr), nbytes, self), device=self.device)

    def __del__(self) -> None:
        with suppress(Exception):
            cu = _driver()
            for offset, chunk in self.maps:
                cu.cuMemUnmap(int(self.ptr) + offset, chunk.nbytes)
            cu.cuMemAddressFree(self.ptr, self.nbytes)


class _ArrayInterface:
    def __init__(self, ptr: int, nbytes: int, owner: Region) -> None:
        self.owner = owner
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 2}


def chunked_tensor(device: torch.device, nbytes: int) -> tuple[torch.Tensor, list[Chunk]]:
    """A uint8 device tensor of ``nbytes`` backed by one chunk per granule, and those chunks in address order."""
    region = Region(device, nbytes)
    chunks = [Chunk(device, region.granule) for _ in range(region.nbytes // region.granule)]
    for i, chunk in enumerate(chunks):
        region.map(i * region.granule, chunk)
    return region.tensor(nbytes), chunks


__all__ = ["Chunk", "Region", "chunked_tensor", "granularity"]
