"""min_p: tokens below min_p x the most likely one are never sampled; the param rides from the
request (or the server default) through SamplingParams into the batch sampler."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler, sample_impl
from freetoken.server.generation import resolve_sampling

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _logits(rows: int) -> torch.Tensor:
    probs = torch.tensor([0.50, 0.30, 0.10, 0.06, 0.02, 0.02])
    return probs.log().repeat(rows, 1).cuda()


@cuda
def test_min_p_never_samples_below_the_floor_and_keeps_the_rest():
    rows = 20000
    t = torch.ones(rows, device="cuda")
    base = sample_impl(_logits(rows), t, None, None)
    assert (base >= 4).float().mean() > 0.02
    cut = sample_impl(_logits(rows), t, None, None, torch.full((rows,), 0.1, device="cuda"))
    assert int((cut >= 4).sum()) == 0
    share = torch.bincount(cut, minlength=6)[:4].float() / rows
    assert torch.allclose(share, torch.tensor([0.5, 0.3, 0.1, 0.06]).cuda() / 0.96, atol=0.015)


@cuda
def test_min_p_composes_with_top_k_and_top_p():
    rows = 20000
    t = torch.ones(rows, device="cuda")
    k = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((rows,), 0.95, device="cuda")
    cut = sample_impl(_logits(rows), t, k, p, torch.full((rows,), 0.1, device="cuda"))
    assert int((cut >= 4).sum()) == 0


def _batch(*params: SamplingParams, verify: bool = False):
    return SimpleNamespace(reqs=[SimpleNamespace(sampling_params=p) for p in params], verify=verify)


@cuda
def test_the_sampler_builds_a_min_p_row_per_sampled_token():
    s = Sampler(torch.device("cuda"), 64)
    args = s.prepare(_batch(SamplingParams(temperature=1.0, min_p=0.05), SamplingParams(temperature=1.0),
                            verify=True))
    assert args.min_p.tolist() == pytest.approx([0.05, 0.05, 0.0, 0.0])
    assert s.prepare(_batch(SamplingParams(temperature=1.0))).min_p is None
    assert s.prepare(_batch(SamplingParams(temperature=0.0, min_p=0.05))).temperatures is None


def _resolve(**kw):
    base = {"temperature": None, "top_k": None, "top_p": None, "max_tokens": None, "ignore_eos": False}
    base.update(kw)
    return resolve_sampling(**base)


def test_min_p_resolves_from_the_request_then_the_defaults():
    assert _resolve(model_sampling={}).min_p == 0.0
    assert _resolve(model_sampling={"min_p": 0.05}).min_p == 0.05
    assert _resolve(model_sampling={"min_p": 0.05}, min_p=0.0).min_p == 0.0
    assert _resolve(model_sampling={}, min_p=0.2).min_p == 0.2
    with pytest.raises(ValueError, match="min_p"):
        _resolve(model_sampling={}, min_p=1.5)


def test_the_checkpoint_generation_config_can_carry_min_p(tmp_path):
    from freetoken.utils.hf import load_generation_sampling

    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 0.7, "top_p": 0.9, "min_p": 0.07}))
    assert load_generation_sampling(str(tmp_path))["min_p"] == 0.07


def test_openai_requests_carry_min_p():
    from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
    from freetoken.server.openai_api import _resolve_sampling

    chat = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], min_p=0.2)
    comp = CompletionRequest(model="m", prompt="hi", min_p=0.3)
    assert _resolve_sampling(chat, {}).min_p == 0.2
    assert _resolve_sampling(comp, {"min_p": 0.05}).min_p == 0.3
