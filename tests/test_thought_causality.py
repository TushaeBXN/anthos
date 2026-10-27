"""
Thought-token causality: with the default mask, thoughts read the whole sequence,
so changing a LATER token changes the logits at EARLIER positions. With
cfg.thought_prefix_only=True, thoughts only read the prompt, so every position
after the prompt is strictly causal.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthos.main import Anthos, AnthosConfig

T = 12
TOL = 1e-6


def _cfg(attn_type: str, prefix_only: bool) -> AnthosConfig:
    kw = dict(
        vocab_size=100, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64,
        max_loop_iters=4, prelude_layers=1, coda_layers=1, n_thought_tokens=4,
        attn_type=attn_type, n_experts=4, n_shared_experts=1, n_experts_per_tok=2,
        expert_dim=32, thought_prefix_only=prefix_only,
        moe_fix_weight_order=prefix_only,     # both fixes are needed for strict causality
    )
    if attn_type == "mla":
        kw.update(kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=8,
                  qk_nope_head_dim=8, v_head_dim=8)
    return AnthosConfig(**kw)


def _model(attn_type: str, prefix_only: bool) -> Anthos:
    torch.manual_seed(0)
    return Anthos(_cfg(attn_type, prefix_only)).eval()


def _change(ids: torch.Tensor, pos: int) -> torch.Tensor:
    out = ids.clone()
    out[:, pos] = (out[:, pos] + 1) % 100
    return out


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_default_mode_still_leaks_and_ignores_prefix_len(attn_type):
    """Characterises the old behaviour so the flag's effect is explicit."""
    m = _model(attn_type, prefix_only=False)
    ids = torch.randint(0, 100, (1, T))
    with torch.no_grad():
        a = m(ids, n_loops=4)
        b = m(_change(ids, T - 1), n_loops=4)
        c = m(ids, n_loops=4, prefix_len=3)          # ignored when the flag is off
    assert (a[:, :-1] - b[:, :-1]).abs().max() > TOL   # leak exists in default mode
    assert torch.equal(a, c)


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
@pytest.mark.parametrize("prefix", [0, 4, 8])
def test_prefix_mode_is_strictly_causal_after_prompt(attn_type, prefix):
    m = _model(attn_type, prefix_only=True)
    ids = torch.randint(0, 100, (2, T))
    for pos in range(max(prefix, 1), T):               # change any response token
        with torch.no_grad():
            a = m(ids, n_loops=4, prefix_len=prefix)
            b = m(_change(ids, pos), n_loops=4, prefix_len=prefix)
        assert (a[:, :pos] - b[:, :pos]).abs().max() <= TOL, f"leak from pos {pos}"


def test_prefix_mode_per_sample_prefix_lengths():
    m = _model("gqa", prefix_only=True)
    ids = torch.randint(0, 100, (2, T))
    pl = torch.tensor([3, 7])
    with torch.no_grad():
        a = m(ids, n_loops=4, prefix_len=pl)
        for b_idx, p in enumerate(pl.tolist()):
            alt = ids.clone()
            alt[b_idx, T - 1] = (alt[b_idx, T - 1] + 1) % 100
            b = m(alt, n_loops=4, prefix_len=pl)
            assert (a[b_idx, : T - 1] - b[b_idx, : T - 1]).abs().max() <= TOL


def test_prompt_still_reaches_the_response():
    """Thoughts must still carry prompt information (the feature is not disabled)."""
    m = _model("gqa", prefix_only=True)
    ids = torch.randint(0, 100, (1, T))
    with torch.no_grad():
        a = m(ids, n_loops=4, prefix_len=6)
        b = m(_change(ids, 1), n_loops=4, prefix_len=6)
    assert (a[:, 8:] - b[:, 8:]).abs().max() > TOL


def test_prefix_mode_requires_prefix_len():
    m = _model("gqa", prefix_only=True)
    with pytest.raises(ValueError):
        m(torch.randint(0, 100, (1, T)), n_loops=2)


def test_prefix_mode_multi_token_decode_is_rejected():
    m = _model("gqa", prefix_only=True)
    ids = torch.randint(0, 100, (1, 5))
    cache = {}
    with torch.no_grad():
        m(ids, n_loops=2, kv_cache=cache, prefix_len=5)
        with pytest.raises(NotImplementedError):
            m(ids[:, -2:], n_loops=2, kv_cache=cache, start_pos=4, prefix_len=5)


def test_moe_weight_order_fix_makes_each_token_independent():
    """Legacy dispatch pairs outputs with another token's router weights."""
    def run(fix):
        cfg = _cfg("gqa", False)
        cfg.moe_fix_weight_order = fix
        torch.manual_seed(0)
        ffn = Anthos(cfg).recurrent.block.ffn.eval()
        torch.manual_seed(1)
        x = torch.randn(1, 16, 64)
        x2 = x.clone()
        x2[:, -1] += 1.0
        with torch.no_grad():
            a, _ = ffn(x)
            b, _ = ffn(x2)
        return (a - b)[:, :15].abs().max().item()

    assert run(False) > TOL    # legacy: other tokens' rows move
    assert run(True) <= TOL    # fixed: only the changed token moves


def test_prefix_mode_rejects_cross_token_routers():
    cfg = _cfg("gqa", True)
    cfg.use_expert_choice = True
    with pytest.raises(ValueError):
        Anthos(cfg)


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_cached_decode_matches_full_forward(attn_type):
    """Prefill + one-token decode steps must reproduce a single full forward pass."""
    m = _model(attn_type, prefix_only=True)
    ids = torch.randint(0, 100, (1, T))
    prompt = 5
    with torch.no_grad():
        full = m(ids, n_loops=4, prefix_len=prompt)
        cache = {}
        pre = m(ids[:, :prompt], n_loops=4, kv_cache=cache, prefix_len=prompt)
        steps = [pre]
        for i in range(prompt, T):
            steps.append(m(ids[:, i:i + 1], n_loops=4, kv_cache=cache, start_pos=i, prefix_len=prompt))
        cached = torch.cat(steps, dim=1)
    assert (full - cached).abs().max() <= 1e-5


def test_generate_runs_in_prefix_mode_and_keeps_prompt():
    m = _model("gqa", prefix_only=True)
    ids = torch.randint(0, 100, (1, 5))
    out = m.generate(ids, max_new_tokens=6, n_loops=4, top_k=10)
    assert out.shape == (1, 11)
    assert torch.equal(out[:, :5], ids)


def test_decode_with_more_loops_than_prefill_is_rejected():
    m = _model("gqa", prefix_only=True)
    ids = torch.randint(0, 100, (1, 5))
    cache = {}
    with torch.no_grad():
        m(ids, n_loops=2, kv_cache=cache, prefix_len=5)
        with pytest.raises(ValueError):
            m(ids[:, -1:], n_loops=3, kv_cache=cache, start_pos=5, prefix_len=5)


@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
def test_fast_thought_decode_matches_reference_path(attn_type):
    """Feeding only the new token must equal the slow path that re-sends the thought rows."""
    ids = torch.randint(0, 100, (2, T))
    prompt = 5

    def run(fast):
        cfg = _cfg(attn_type, True)
        cfg.fast_thought_decode = fast
        torch.manual_seed(0)
        m = Anthos(cfg).eval()
        cache = {}
        with torch.no_grad():
            outs = [m(ids[:, :prompt], n_loops=4, kv_cache=cache, prefix_len=prompt)]
            for i in range(prompt, T):
                outs.append(m(ids[:, i:i + 1], n_loops=4, kv_cache=cache, start_pos=i, prefix_len=prompt))
        return torch.cat(outs, dim=1)

    assert (run(True) - run(False)).abs().max() <= 1e-5
