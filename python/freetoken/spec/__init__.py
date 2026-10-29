"""Speculative decoding: SpecRunner (the target side) and one Proposer (the draft side) per method."""

from freetoken.utils import Registry

# method name -> "module:Class" of its Proposer, loaded on use so this package imports no method's code
PROPOSERS = Registry[str]("Speculative proposer")
PROPOSERS.register("mtp")("freetoken.models.qwen3_5_moe.mtp_proposer:Qwen3_5MTPProposer")

__all__ = ["PROPOSERS"]
