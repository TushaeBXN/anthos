import torch

from regular import PRESETS, Regular
from regular.model import apply_rope, rope_table


def _tiny():
    torch.manual_seed(0)
    return Regular(PRESETS["tiny"]).eval()


def test_parameter_count_matches_the_formula_for_every_preset():
    for cfg in PRESETS.values():
        if cfg.dim <= 128:
            assert Regular(cfg).num_params() == Regular.expected_params(cfg)
    assert Regular.expected_params(PRESETS["138m"]) == 138_431_232


def test_output_shape_and_tied_embedding():
    m = _tiny()
    out = m(torch.randint(0, 1024, (3, 17)))
    assert out.shape == (3, 17, 1024)
    assert not any(n.endswith("lm_head.weight") for n, _ in m.named_parameters())     # output layer is the embedding


def test_it_is_causal_a_later_token_cannot_change_earlier_logits():
    m = _tiny()
    a = torch.randint(0, 1024, (2, 20))
    b = a.clone(); b[:, 12:] = torch.randint(0, 1024, (2, 8))
    with torch.no_grad():
        assert torch.allclose(m(a)[:, :12], m(b)[:, :12], atol=1e-5)
        assert not torch.allclose(m(a)[:, 12:], m(b)[:, 12:], atol=1e-5)


def test_rope_keeps_vector_length_and_depends_on_position():
    cos, sin = rope_table(16, 32, 10000.0)
    x = torch.randn(1, 2, 8, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    assert torch.allclose(x[..., 0, :], y[..., 0, :], atol=1e-6)                      # position 0 is not rotated
    assert not torch.allclose(x[..., 5, :], y[..., 5, :], atol=1e-3)


def test_it_can_overfit_one_batch():
    torch.manual_seed(0)
    m = Regular(PRESETS["tiny"])
    x = torch.randint(0, 1024, (4, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < first * 0.2


def test_a_checkpoint_round_trips():
    m = _tiny()
    m2 = Regular(PRESETS["tiny"]).eval()
    m2.load_state_dict(m.state_dict())
    x = torch.randint(0, 1024, (2, 9))
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))
