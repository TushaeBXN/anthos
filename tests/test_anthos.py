"""
Anthos — Test suite
Run with: python -m pytest tests/ -v
"""

import pytest
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anthos.main import (
    AnthosConfig, Anthos, ThoughtTokenPool,
    LTIInjection, _anthos_causal_mask, _anthos_rope_freqs,
)


@pytest.fixture
def cfg():
    return AnthosConfig(
        vocab_size=256, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=32, max_loop_iters=6, prelude_layers=1, coda_layers=1,
        attn_type="gqa", n_experts=4, n_shared_experts=1,
        n_experts_per_tok=2, expert_dim=16, lora_rank=4,
        n_thought_tokens=8, moe_aux_coef=1e-2, act_aux_coef=1e-3,
    )


@pytest.fixture
def model(cfg):
    return Anthos(cfg)


@pytest.fixture
def ids(cfg):
    return torch.randint(0, cfg.vocab_size, (2, 8))


class TestForward:
    def test_output_shape(self, model, cfg, ids):
        logits = model(ids, n_loops=4)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_thought_tokens_not_in_output(self, model, cfg, ids):
        logits = model(ids, n_loops=4)
        # Output should be seq_len only, NOT n_thought + seq_len
        assert logits.shape[1] == 8, "Thought tokens leaked into output"

    def test_aux_loss_scalar(self, model, ids):
        logits, aux = model(ids, n_loops=4, return_aux=True)
        assert aux.numel() == 1
        assert aux.item() >= 0

    def test_backward(self, model, cfg, ids):
        labels = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, aux = model(ids, n_loops=4, return_aux=True)
        ce   = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), labels[:, 1:].reshape(-1))
        loss = ce + aux
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        assert all(not g.isnan().any() for g in grads), "NaN gradients"

    def test_no_aux_flag(self, model, ids):
        result = model(ids, n_loops=4, return_aux=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape[1] == 8


class TestCausalMask:
    def test_thought_rows_full_attention(self, cfg):
        mask = _anthos_causal_mask(cfg.n_thought_tokens, 8, torch.device("cpu"))
        thought_rows = mask[0, 0, :cfg.n_thought_tokens, :]
        assert thought_rows.eq(0).all(), "Thought rows must be full attention (all zeros)"

    def test_seq_to_thought_open(self, cfg):
        mask = _anthos_causal_mask(cfg.n_thought_tokens, 8, torch.device("cpu"))
        seq_to_thought = mask[0, 0, cfg.n_thought_tokens:, :cfg.n_thought_tokens]
        assert seq_to_thought.eq(0).all(), "Sequence must attend to all thought tokens"

    def test_seq_to_seq_causal(self, cfg):
        mask = _anthos_causal_mask(cfg.n_thought_tokens, 8, torch.device("cpu"))
        n = cfg.n_thought_tokens
        # Position (n, n+1) should be -inf (tok 0 cannot see tok 1)
        assert mask[0, 0, n, n + 1].item() == float("-inf")
        # Position (n+1, n) should be 0 (tok 1 can see tok 0)
        assert mask[0, 0, n + 1, n].item() == 0.0

    def test_mask_shape(self, cfg):
        mask = _anthos_causal_mask(cfg.n_thought_tokens, 8, torch.device("cpu"))
        expected = (1, 1, cfg.n_thought_tokens + 8, cfg.n_thought_tokens + 8)
        assert mask.shape == expected


class TestStability:
    def test_hidden_state_norm_bounded(self, model, cfg, ids):
        model.eval()
        norms = []
        _orig = model.recurrent.h_norm.forward
        def _record(x):
            out = _orig(x)
            norms.append(out.norm().item())
            return out
        model.recurrent.h_norm.forward = _record

        with torch.no_grad():
            model(ids, n_loops=6)

        ratio = max(norms) / (min(norms) + 1e-9)
        assert ratio < 5.0, f"Hidden state exploded: max/min ratio = {ratio:.2f}"

    def test_lti_spectral_radius_seq(self, model):
        A = model.recurrent.seq_injection.get_A()
        assert A.max().item() < 1.0, "Sequence LTI spectral radius >= 1 — unstable"

    def test_lti_spectral_radius_thought(self, model):
        A = model.recurrent.thought_injection.get_A()
        assert A.max().item() < 1.0, "Thought LTI spectral radius >= 1 — unstable"

    def test_no_nan_in_output(self, model, ids):
        logits = model(ids, n_loops=6)
        assert not logits.isnan().any(), "NaN in logits"
        assert not logits.isinf().any(), "Inf in logits"


class TestGeneration:
    def test_generate_shape(self, model, cfg):
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=8, n_loops=2)
        assert out.shape == (1, 12)

    def test_generate_extends_prompt(self, model, cfg):
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=5, n_loops=2)
        # First 4 tokens should match the prompt
        assert (out[:, :4] == prompt).all()


class TestLearning:
    """Verify the model actually learns — not just that it runs."""

    def test_overfit_single_batch(self):
        """
        Model must reduce loss by >80% on a fixed batch in 50 steps.
        This is the definitive proof that gradients flow correctly and
        the architecture can learn. If this fails, something is broken
        at a fundamental level — not just a shape or mask issue.
        """
        import math
        torch.manual_seed(42)
        cfg = AnthosConfig(
            vocab_size=64, dim=128, n_heads=4, n_kv_heads=2,
            max_seq_len=32, max_loop_iters=4, prelude_layers=1, coda_layers=1,
            attn_type="gqa", n_experts=4, n_shared_experts=1,
            n_experts_per_tok=2, expert_dim=32, lora_rank=4, n_thought_tokens=4,
        )
        model = Anthos(cfg)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        fixed = torch.randint(0, 64, (2, 32))

        first_loss = None
        for step in range(50):
            opt.zero_grad()
            logits, aux = model(fixed[:, :-1], n_loops=4, return_aux=True)
            loss = F.cross_entropy(logits.reshape(-1, 64), fixed[:, 1:].reshape(-1)) + aux
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        drop_pct = (first_loss - loss.item()) / first_loss * 100
        assert drop_pct > 80, f"Model not learning: only {drop_pct:.1f}% loss reduction"



    def test_init_batch_shape(self, cfg):
        pool = ThoughtTokenPool(cfg.n_thought_tokens, cfg.dim)
        out  = pool.init_batch(3, torch.device("cpu"))
        assert out.shape == (3, cfg.n_thought_tokens, cfg.dim)

    def test_batches_are_independent(self, cfg):
        pool = ThoughtTokenPool(cfg.n_thought_tokens, cfg.dim)
        a = pool.init_batch(2, torch.device("cpu"))
        b = pool.init_batch(2, torch.device("cpu"))
        # Modifying a should not affect b (contiguous clone)
        a[0, 0, 0] = 999.0
        assert b[0, 0, 0].item() != 999.0
