"""Recall probe plumbing: the planted answer is right, and the memory state genuinely carries information."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import memory_recall as M
from anthos.main import Anthos, AnthosConfig


def tiny(bank=True):
    cfg = AnthosConfig(vocab_size=8192, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128, max_loop_iters=2,
                       prelude_layers=1, coda_layers=1, attn_type="gqa", n_experts=4, n_shared_experts=1,
                       n_experts_per_tok=2, expert_dim=32, lora_rank=4, n_thought_tokens=4,
                       thought_prefix_only=True, moe_fix_weight_order=True, use_memory_bank=bank)
    torch.manual_seed(0)
    return Anthos(cfg).eval()


def test_batches_plant_the_right_answer():
    gen = torch.Generator().manual_seed(1)
    fact, fillers, query, target = M.make_batch(32, 4, 2, 16, gen)
    assert fact.shape == (32, 16) and len(fillers) == 2 and query.shape == (32, 2)
    for b in range(32):
        keys, vals = fact[b, 1::4].tolist(), fact[b, 3::4].tolist()
        assert len(set(keys)) == 4                                   # keys are distinct, so the question is unambiguous
        assert target[b].item() == vals[keys.index(query[b, 1].item())]
    assert (query[:, 0] == M.ASK).all() and (fact[:, 0::4] == M.FACT).all() and (fact[:, 2::4] == M.VAL).all()


def _logits(model, fact_tok, mode):
    fact = torch.tensor([[M.FACT, M.KEY0 + 1, M.VAL, M.VAL0 + fact_tok]])
    batch = (fact, [torch.randint(M.FILL0, M.FILL1, (1, 8), generator=torch.Generator().manual_seed(5))],
             torch.tensor([[M.ASK, M.KEY0 + 1]]), torch.tensor([M.VAL0 + fact_tok]))
    with torch.no_grad():
        return M.answer_logits(model, batch, mode, n_loops=2)


def _state(model, fact_tok):
    fact = torch.tensor([[M.FACT, M.KEY0 + 1, M.VAL, M.VAL0 + fact_tok]])
    with torch.no_grad():
        model(fact, n_loops=2, return_aux=True, prefix_len=4)
    s = model._last_memory_state
    return s.keys.clone(), s.values.clone()


def test_the_stored_state_depends_on_the_fact():
    # Information does reach the bank. (How much of it the read path passes on is what training has to learn: at
    # initialization the effect on the answer is ~1e-7, which is the "never taught to use it" situation.)
    model = tiny()
    k3, v3 = _state(model, 3)
    k3b, v3b = _state(model, 3)
    k40, v40 = _state(model, 40)
    assert torch.equal(k3, k3b) and torch.equal(v3, v3b)
    assert (v3 - v40).abs().max() > 1e-3 and (k3 - k40).abs().max() > 1e-3


def test_a_reset_state_cannot_see_the_fact():
    model = tiny()
    a, b = _logits(model, 3, "reset"), _logits(model, 40, "reset")
    assert torch.allclose(a, b, atol=1e-6)


def test_with_the_bank_removed_nothing_carries():
    model = tiny(bank=False)
    a, b = _logits(model, 3, "carry"), _logits(model, 40, "carry")
    assert torch.allclose(a, b, atol=1e-5)


def test_training_step_is_finite_and_reaches_the_bank():
    model = tiny().train()
    gen = torch.Generator().manual_seed(2)
    batch = M.make_batch(8, 2, 1, 8, gen)
    loss = torch.nn.functional.cross_entropy(M.answer_logits(model, batch, "carry", 2), batch[3])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.recurrent.memory_bank.parameters())
