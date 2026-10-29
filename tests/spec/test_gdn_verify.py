"""The GDN verify kernels: the fused recurrence's verify mode stashes each row's inputs, and gdn_state_replay over the first n rows reproduces decode mode bitwise for every n.

The layer-level verify (outputs, live state, commit) is checked against the reference in tests/layers/test_gdn.py.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NK, NV, HD = 2, 4, 32
STEPS = 5  # verify width: k drafts + 1


@cuda
# the second geometry launches more layer x row x head programs than one grid axis holds (48 * 29 * 48 > 65535)
@pytest.mark.parametrize("layers,bs,h,hv,dim", [(2, 3, NK, NV, HD), (48, 29, 16, 48, 16)], ids=["small", "past-grid-axis-limit"])
def test_kernel_verify_stash_replay_matches_decode_bitwise(layers, bs, h, hv, dim):
    """The fused kernel's verify mode (disable_state_update + input stash) followed by
    gdn_state_replay over the first n tokens must reproduce decode mode BITWISE on
    identical per-token inputs, for every n, across layers and with verify rows that
    differ from the live slots; the verify itself must leave the live slab untouched."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused
    from freetoken.kernel.fla import gdn_state_replay

    torch.manual_seed(0)
    dev = torch.device("cuda")
    steps, kd, vd = STEPS, dim, dim
    slots = [(i + 2) % bs for i in range(bs)]  # verify row i commits into live slot slots[i]

    def r(*s, dt=torch.bfloat16, sc=0.5):
        return (torch.randn(*s, device=dev) * sc).to(dt)

    q, k, v = r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, hv, vd)
    a, b = r(layers, bs * steps, hv), r(layers, bs * steps, hv)
    a_log, dt_bias = r(layers, hv, dt=torch.float32, sc=0.1), r(layers, hv, dt=torch.float32, sc=0.1)
    state0 = torch.randn(layers, bs + 1, hv, kd, vd, device=dev) * 0.1
    slot_t = torch.tensor(slots, dtype=torch.int32, device=dev)
    common = dict(softplus_beta=1.0, softplus_threshold=20.0, scale=kd ** -0.5,
                  use_qk_l2norm_in_kernel=True, initial_state_indices=slot_t)

    per_step = []  # per_step[t][l] = live slab after tokens 0..t
    st_decode = state0.clone()
    for t in range(steps):
        rows = torch.arange(bs, device=dev) * steps + t
        for l in range(layers):
            fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l][rows], q=q[l][:, rows], k=k[l][:, rows],
                  v=v[l][:, rows], b=b[l][rows], initial_state_source=st_decode[l],
                  cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev), **common)
        per_step.append(st_decode.clone())

    stash = tuple(torch.zeros(layers, bs, steps, *shape, device=dev)
                  for shape in ((h, kd), (hv, vd), (hv,), (hv,)))
    st_verify = state0.clone()
    for l in range(layers):
        fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l], q=q[l], k=k[l], v=v[l], b=b[l],
              initial_state_source=st_verify[l],
              cu_seqlens=torch.arange(bs + 1, dtype=torch.int64, device=dev) * steps,
              disable_state_update=True, input_stash=tuple(t[l] for t in stash), **common)
    assert torch.equal(st_verify, state0)  # live slab untouched

    for n in range(1, steps + 1):
        st = state0.clone()
        meta = torch.tensor([list(range(bs)), slots, [n] * bs], dtype=torch.int32, device=dev)
        gdn_state_replay(st, stash, meta)
        assert torch.equal(st[:, slots], per_step[n - 1][:, slots]), f"n={n}"
        assert torch.equal(st[:, bs], state0[:, bs])  # the slot no row commits to


@cuda
def test_kernel_verify_one_row_per_call_matches_all_rows_bitwise():
    """A staggered verify runs row t of every request in its own call, replaying the t rows earlier calls stashed;
    each call's output and stash must equal the all-rows verify's bitwise, and the live slab stays untouched."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused

    torch.manual_seed(1)
    dev = torch.device("cuda")
    bs, steps, h, hv, kd = 3, 4, NK, NV, HD

    def r(*s, dt=torch.bfloat16, sc=0.5):
        return (torch.randn(*s, device=dev) * sc).to(dt)

    q, k, v = r(1, bs * steps, h, kd), r(1, bs * steps, h, kd), r(1, bs * steps, hv, kd)
    a, b = r(bs * steps, hv), r(bs * steps, hv)
    a_log, dt_bias = r(hv, dt=torch.float32, sc=0.1), r(hv, dt=torch.float32, sc=0.1)
    state0 = torch.randn(bs + 1, hv, kd, kd, device=dev) * 0.1
    common = {
        "A_log": a_log, "dt_bias": dt_bias, "softplus_beta": 1.0, "softplus_threshold": 20.0, "scale": kd ** -0.5,
        "use_qk_l2norm_in_kernel": True, "disable_state_update": True,
        "initial_state_indices": torch.tensor([2, 0, 3], dtype=torch.int32, device=dev),
    }

    def stash():
        return tuple(torch.zeros(bs, steps, *shape, device=dev) for shape in ((h, kd), (hv, kd), (hv,), (hv,)))

    want_stash, st = stash(), state0.clone()
    want = fused(a=a, q=q, k=k, v=v, b=b, initial_state_source=st, input_stash=want_stash,
                 cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev) * steps, **common)
    got_stash = stash()
    for t in range(steps):
        rows = torch.arange(bs, device=dev) * steps + t
        got = fused(a=a[rows], q=q[:, rows], k=k[:, rows], v=v[:, rows], b=b[rows], initial_state_source=st,
                    input_stash=got_stash, prior_steps=t, cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev), **common)
        assert torch.equal(got, want[:, rows]), f"row {t}"
    assert all(torch.equal(g, w) for g, w in zip(got_stash, want_stash))
    assert torch.equal(st, state0)
