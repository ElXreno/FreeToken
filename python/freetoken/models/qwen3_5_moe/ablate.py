"""Remove one direction from the residual stream wherever a sublayer reads it.

This is the runtime form of the directional ablation in Arditi et al. (arXiv:2406.11717):
subtracting ``alpha * r r^T x`` at every read point is equivalent to orthogonalizing every matrix
that writes to the stream, but it needs no requantization of NVFP4 weights and keeps the
direction and its strength startup options instead of a new checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

_CUT, _ADD_CUT, _CUT_SUM = 0, 1, 2


@triton.jit
def _cut_rows(x_ptr, y_ptr, r_ptr, alpha, stride_x, stride_y,
              H: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr):
    """One program per token row, written back into ``x``.

    MODE 0: x <- x - a (x.r) r.  MODE 1: x <- s - a (s.r) r with s = x + y.
    MODE 2: x <- x - a (s.r) r, y untouched.  The sum is rounded to the storage dtype before the
    dot product, the same rounding the fused add+norm applies to the stream it stores.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if MODE == 0:
        s = x
    else:
        y = tl.load(y_ptr + row * stride_y + offs, mask=mask, other=0.0).to(tl.float32)
        s = (x + y).to(x_ptr.dtype.element_ty).to(tl.float32)
    d = tl.sum(s * r, axis=0) * alpha
    if MODE == 2:
        out = x - d * r
    else:
        out = s - d * r
    tl.store(x_ptr + row * stride_x + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


class DirectionAblation:
    """``cut(x) = x - alpha * (x . r) r`` for a unit vector ``r`` picked from a per-layer table.

    The table is a safetensors file holding one tensor, ``direction``, of shape
    ``[layers, hidden]``; a row is chosen once at startup and applied at every layer. With
    ``alpha`` below 1 the projection is not idempotent, so two cuts at the same point of the
    stream compound; callers rely on that and must not deduplicate them.

    The in-place forms run one Triton program per row on CUDA: a prefill chunk's stream is
    [tokens, hidden], and anything that materializes the projection or wakes cuBLAS for the
    dot product costs a chunk-sized buffer the fused path never needed.
    """

    def __init__(self, path: str, layer: int, alpha: float, hidden_size: int) -> None:
        from safetensors.torch import load_file

        table = load_file(path)
        if "direction" not in table:
            raise ValueError(f"{path}: no 'direction' tensor (found {sorted(table)})")
        directions = table["direction"].float()
        if directions.ndim != 2 or directions.shape[1] != hidden_size:
            raise ValueError(
                f"{path}: direction has shape {tuple(directions.shape)}, want [layers, {hidden_size}]"
            )
        if not 0 <= layer < directions.shape[0]:
            raise ValueError(f"ablate layer {layer} is outside the {directions.shape[0]} rows of {path}")
        row = directions[layer]
        self._unit = row / row.norm().clamp(min=1e-6)
        self.alpha = float(alpha)
        # placed on first use: the model is built before it knows its device
        self._placed: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @classmethod
    def from_config(cls, config: ModelConfig) -> DirectionAblation | None:
        if not config.ablate_direction:
            return None
        return cls(config.ablate_direction, config.ablate_layer, config.ablate_alpha, config.hidden_size)

    def _vector(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        r = self._placed.get(key)
        if r is None:
            r = self._placed[key] = self._unit.to(x.device, x.dtype)
        return r

    def _launch(self, x: torch.Tensor, y: torch.Tensor | None, mode: int) -> torch.Tensor:
        assert x.dim() == 2 and x.stride(1) == 1, (tuple(x.shape), x.stride())
        other = x if y is None else y
        assert other.shape == x.shape and other.stride(1) == 1
        if x.shape[0]:
            h = x.shape[1]
            _cut_rows[(x.shape[0],)](
                x, other, self._vector(x), self.alpha, x.stride(0), other.stride(0),
                H=h, BLOCK=triton.next_power_of_2(h), MODE=mode, num_warps=4,
            )
        return x

    def component(self, x: torch.Tensor) -> torch.Tensor:
        r = self._vector(x)
        return self.alpha * ((x @ r).unsqueeze(-1) * r)

    def cut(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.component(x)

    def cut_(self, x: torch.Tensor) -> torch.Tensor:
        """``cut`` written back into ``x``."""
        if x.is_cuda:
            return self._launch(x, None, _CUT)
        return x.copy_(self.cut(x))

    def add_cut_(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``x <- cut(x + y)``: the stream accumulates in ``x``, as in the fused add+norm."""
        if x.is_cuda:
            return self._launch(x, y, _ADD_CUT)
        return x.copy_(self.cut(x + y))

    def cut_sum_(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``x <- x - component(x + y)``, so ``x + y`` comes out cut and ``y`` stays as it was."""
        if x.is_cuda:
            return self._launch(x, y, _CUT_SUM)
        return x.sub_(self.component(x + y))


__all__ = ["DirectionAblation"]
