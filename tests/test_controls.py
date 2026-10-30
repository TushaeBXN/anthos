"""Control and ablation options: ACT output modes and the dense-FFN variant of the looped block."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anthos.main import Anthos, AnthosConfig


def make_cfg(**kw):
    base = dict(
        vocab_size=256, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=32, max_loop_iters=4,
        prelude_layers=1, coda_layers=1, attn_type="gqa", n_experts=4, n_shared_experts=1,
        n_experts_per_tok=2, expert_dim=16, lora_rank=4, n_thought_tokens=4,
    )
    base.update(kw)
    return AnthosConfig(**base)


def step(cfg, n_loops=4, seed=0, halt_bias=None):
    """One forward/backward. halt_bias=-8 makes the halting head output ~0, the regime the trained models are in."""
    torch.manual_seed(seed)
    model = Anthos(cfg)
    if halt_bias is not None:
        model.recurrent.act.halt.weight.data.zero_()
        model.recurrent.act.halt.bias.data.fill_(halt_bias)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, aux = model(ids[:, :-1], n_loops=n_loops, return_aux=True)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)) + aux
    loss.backward()
    return model, logits, loss


def test_defaults_are_the_trained_configuration():
    cfg = AnthosConfig()
    assert (cfg.act_mode, cfg.ffn_type, cfg.dense_ffn_dim) == ("standard", "moe", 0)


@pytest.mark.parametrize("mode", ["standard", "remainder", "last"])
def test_act_modes_train(mode):
    model, logits, loss = step(make_cfg(act_mode=mode))
    assert torch.isfinite(loss) and torch.isfinite(logits).all()
    assert model.embed.weight.grad is not None and torch.isfinite(model.embed.weight.grad).all()


def test_act_modes_differ_when_tokens_do_not_halt():
    outs = {}
    for mode in ("standard", "remainder", "last"):
        model, logits, _ = step(make_cfg(act_mode=mode), halt_bias=-8.0)
        outs[mode] = logits.detach()
        assert model.halt_probs.sum(-1).max() < 0.99         # nobody reached the threshold
    assert not torch.allclose(outs["standard"], outs["remainder"], atol=1e-3)
    assert not torch.allclose(outs["standard"], outs["last"], atol=1e-3)


def test_remainder_with_a_silent_halting_head_equals_the_last_loop():
    # p ~ 0 everywhere: remainder mode puts ~all weight on the final loop, which is exactly what "last" returns.
    _, rem, _ = step(make_cfg(act_mode="remainder"), halt_bias=-12.0)
    _, last, _ = step(make_cfg(act_mode="last"), halt_bias=-12.0)
    assert torch.allclose(rem.detach(), last.detach(), atol=1e-2)


def test_last_mode_runs_every_loop_and_logs_no_halting():
    model, _, _ = step(make_cfg(act_mode="last"))
    assert model.halt_probs.abs().sum() == 0


def test_dense_ffn_replaces_the_moe_and_trains():
    cfg = make_cfg(ffn_type="dense")
    model, logits, loss = step(cfg)
    ffn = model.recurrent.block.ffn
    assert not model.recurrent.block.use_moe and not hasattr(ffn, "routed_experts")
    # parameter-matched hidden size = routed experts + shared experts at topk width
    assert ffn.gate.out_features == 4 * 16 + 1 * 16 * 2
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


def test_dense_ffn_explicit_width():
    model = Anthos(make_cfg(ffn_type="dense", dense_ffn_dim=48))
    assert model.recurrent.block.ffn.gate.out_features == 48


def test_plain_control_one_loop_no_thoughts_dense():
    cfg = make_cfg(ffn_type="dense", max_loop_iters=1, n_thought_tokens=0)
    model, logits, loss = step(cfg, n_loops=1)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


def test_bad_options_are_rejected():
    with pytest.raises(ValueError):
        Anthos(make_cfg(act_mode="nope"))
    with pytest.raises(ValueError):
        Anthos(make_cfg(ffn_type="nope"))


def test_default_param_count_is_unchanged():
    """The trained models must still be exactly representable: proof config at vocab 100352 = 73,722,372."""
    from anthos.configs import get_training_config
    cfg, _ = get_training_config("proof")
    cfg.vocab_size = 100352
    assert sum(p.numel() for p in Anthos(cfg).parameters()) == 73_722_372


def test_new_switches_default_to_the_trained_configuration():
    cfg = AnthosConfig()
    assert (cfg.use_memory_bank, cfg.use_loop_lora, cfg.injection) == (True, True, "lti")


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_memory_bank_switch_removes_it_and_still_trains():
    full, _, _ = step(make_cfg())
    model, logits, loss = step(make_cfg(use_memory_bank=False))
    assert model.recurrent.memory_bank is None and _n_params(model) < _n_params(full)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


def test_loop_lora_switch_removes_it_and_still_trains():
    full, _, _ = step(make_cfg())
    model, logits, loss = step(make_cfg(use_loop_lora=False))
    assert model.recurrent.lora is None and _n_params(model) < _n_params(full)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


def test_plain_injection_has_no_parameters_and_still_trains():
    full, _, _ = step(make_cfg())
    model, logits, loss = step(make_cfg(injection="plain"))
    assert _n_params(model) < _n_params(full)
    assert not list(model.recurrent.seq_injection.parameters()) and not list(model.recurrent.thought_injection.parameters())
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


def test_everything_off_together_trains():
    cfg = make_cfg(use_memory_bank=False, use_loop_lora=False, injection="plain", ffn_type="dense", act_mode="last")
    model, logits, loss = step(cfg)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()


def test_injection_option_is_validated():
    with pytest.raises(ValueError):
        Anthos(make_cfg(injection="nope"))
