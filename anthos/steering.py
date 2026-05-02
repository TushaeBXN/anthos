"""
anthos/steering.py — Activation Steering and Collection for Anthos

Two capabilities:

  1. AnthosSteer — Non-destructive personality injection via forward hooks.
     Uses the "Activation Addition" method: adds a learned direction vector
     to hidden states at a chosen layer, shifting the model's output style
     without retraining a single weight.

  2. ActivationCollector — Attach hooks to capture thought/sequence stream
     activations across loop iterations (needed for SAE training + analysis).

  3. ActivationSteering — Inject SAE feature directions into the thought or
     sequence stream at inference time. Zero weight updates required.

  4. LTIStateSteering — Steers the LTI recurrent hidden state h_t directly
     (more persistent across loop iterations than ActivationSteering).

Steering formula (from Qwen-Scope):
    h' ← h + α · d
where d is a unit-norm SAE feature direction and α is the steering strength.

Usage — Persona steering:
    from anthos.steering import AnthosSteer

    steer = AnthosSteer(model, target="recurrent")
    steer.load_persona("vectors/tars_rogue.pt")
    steer.engage(strength=0.75)

Usage — Collecting activations:
    from anthos.steering import ActivationCollector

    collector = ActivationCollector(model, stream="thought")
    collector.attach()
    with torch.no_grad():
        logits = model(input_ids, n_loops=8)
    acts = collector.flat_activations()   # [N, D]
    collector.detach()

Usage — Steering at inference:
    from anthos.steering import ActivationSteering

    feature_dir = sae.W_dec[feature_id]          # unit-norm direction [D]
    steerer = ActivationSteering(model, stream="thought", n_thought_tokens=16)
    steerer.set_direction(feature_dir, alpha=-8.0)   # negative = suppress
    steerer.attach()
    out = model.generate(input_ids, max_new_tokens=128, n_loops=12)
    steerer.detach()
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Hook target resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_recurrent_blocks(model: nn.Module) -> list[nn.Module]:
    """
    Walk the model graph and return all RecurrentBlock instances.
    Works by class name so it doesn't require importing anthos internals.
    """
    blocks = []
    for module in model.modules():
        if type(module).__name__ == "RecurrentBlock":
            blocks.append(module)
    return blocks


def _find_attention_layers(model: nn.Module) -> list[nn.Module]:
    """Return all attention modules for finer-grained hook placement."""
    layers = []
    for module in model.modules():
        name = type(module).__name__
        if "Attention" in name or "SelfAttn" in name:
            layers.append(module)
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# AnthosSteer — Persona / Activation-Addition Steering
# ─────────────────────────────────────────────────────────────────────────────

class AnthosSteer:
    """
    Activation-addition steering for Anthos models.

    Supported hook targets:
      "recurrent"   — the main AnthosRecurrentBlock (recommended)
      "prelude_N"   — prelude TransformerBlock at index N  (e.g. "prelude_0")
      "coda_N"      — coda TransformerBlock at index N     (e.g. "coda_0")

    Parameters
    ----------
    model      : Anthos instance
    target     : which layer to steer — "recurrent" | "prelude_N" | "coda_N"
    """

    def __init__(self, model: nn.Module, target: str = "recurrent"):
        self.model  = model
        self.target = target
        self.handle = None
        self.vector: torch.Tensor | None = None

    # ── Persona I/O ───────────────────────────────────────────────────────────

    def load_persona(self, path: str | Path) -> None:
        """Load a steering vector from a .pt file."""
        self.vector = torch.load(path, map_location="cpu")
        print(f"[AnthosSteer] Persona loaded from {path}  shape={tuple(self.vector.shape)}")

    def save_persona(self, path: str | Path) -> None:
        """Save the current steering vector to a .pt file."""
        if self.vector is None:
            raise ValueError("No vector loaded — nothing to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.vector, path)
        print(f"[AnthosSteer] Persona saved → {path}")

    # ── Hook helpers ──────────────────────────────────────────────────────────

    def _resolve_layer(self) -> nn.Module:
        """Return the nn.Module to attach the hook to."""
        t = self.target
        if t == "recurrent":
            return self.model.recurrent
        if t.startswith("prelude_"):
            idx = int(t.split("_")[1])
            return self.model.prelude[idx]
        if t.startswith("coda_"):
            idx = int(t.split("_")[1])
            return self.model.coda[idx]
        raise ValueError(
            f"Unknown target '{t}'. Use 'recurrent', 'prelude_N', or 'coda_N'."
        )

    def _make_hook(self, strength: float):
        """
        Build a forward hook that adds the persona vector to hidden states.

        Anthos layer output shapes:
          TransformerBlock  → (hidden, aux_loss)       hidden: (B, T, D)
          RecurrentBlock    → (h_out, moe_aux, act_aux) h_out: (B, T, D)
        """
        vec = self.vector

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                v = vec.to(hidden.device, hidden.dtype)
                if v.dim() == 1:
                    v = v.unsqueeze(0).unsqueeze(0)   # (1, 1, D)
                modified = hidden + v * strength
                return (modified,) + output[1:]
            else:
                v = vec.to(output.device, output.dtype)
                if v.dim() == 1:
                    v = v.unsqueeze(0).unsqueeze(0)
                return output + v * strength

        return hook_fn

    # ── Engage / Disengage ────────────────────────────────────────────────────

    def engage(self, strength: float = 0.75) -> None:
        """
        Attach the persona vector to the model.

        strength : float
            Scaling factor for the vector. 0.0 = no effect, 1.0 = full vector.
            Start around 0.5–0.8; higher values can destabilise generation.
        """
        if self.vector is None:
            raise ValueError("No persona loaded. Call load_persona() first.")
        if self.handle is not None:
            print("[AnthosSteer] Already engaged — disengaging first.")
            self.disengage()

        layer = self._resolve_layer()
        self.handle = layer.register_forward_hook(self._make_hook(strength))
        print(f"[AnthosSteer] Tactical Persona engaged  target={self.target}  strength={strength}")

    def disengage(self) -> None:
        """Remove the hook and return the model to its default behaviour."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            print("[AnthosSteer] Persona disengaged — returning to default mode.")
        else:
            print("[AnthosSteer] Nothing to disengage.")

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.engage()
        return self

    def __exit__(self, *args):
        self.disengage()

    def __repr__(self) -> str:
        status = "engaged" if self.handle else "disengaged"
        vec_shape = tuple(self.vector.shape) if self.vector is not None else None
        return f"AnthosSteer(target={self.target!r}, vector={vec_shape}, status={status})"


# ─────────────────────────────────────────────────────────────────────────────
# ActivationCollector
# ─────────────────────────────────────────────────────────────────────────────

class ActivationCollector:
    """
    Captures activations from the recurrent block output across loop iterations.

    Anthos's recurrent block outputs [B, n_thought + T, D].
    We slice on n_thought_tokens to separate streams.

    Args:
        model:            Anthos model instance
        stream:           "thought" | "sequence" | "both"
        n_thought_tokens: Must match model cfg.n_thought_tokens (default 16)
    """

    def __init__(
        self,
        model: nn.Module,
        stream: str = "thought",
        n_thought_tokens: int = 16,
    ):
        self.model = model
        self.stream = stream
        self.n_thought = n_thought_tokens
        self._hooks: list = []
        self._store: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._loop_counter = 0

    def _make_hook(self) -> Callable:
        collector = self

        def hook(module, input, output):
            # output shape: [B, n_thought + T, D]
            # Capture a detached CPU copy to avoid OOM during long runs
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            n = collector.n_thought
            if collector.stream == "thought":
                captured = act[:, :n, :].detach().cpu()
            elif collector.stream == "sequence":
                captured = act[:, n:, :].detach().cpu()
            else:  # "both"
                captured = act.detach().cpu()

            collector._store[collector._loop_counter].append(captured)
            collector._loop_counter += 1

        return hook

    def attach(self):
        """Register forward hooks on all RecurrentBlock instances."""
        self.detach()
        self._loop_counter = 0
        self._store.clear()

        blocks = _find_recurrent_blocks(self.model)
        if not blocks:
            raise RuntimeError(
                "No RecurrentBlock found in model. "
                "Ensure anthos.main is imported and model is an Anthos instance."
            )

        hook_fn = self._make_hook()
        for block in blocks:
            h = block.register_forward_hook(hook_fn)
            self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get(self) -> dict[int, torch.Tensor]:
        """
        Returns collected activations per loop iteration.
        Shape per entry: [B, T_stream, D]
        (concatenated across all forward calls since last attach/clear)
        """
        return {
            loop: torch.cat(tensors, dim=0)
            for loop, tensors in self._store.items()
        }

    def clear(self):
        self._store.clear()
        self._loop_counter = 0

    def flat_activations(self) -> torch.Tensor:
        """
        Flatten all collected activations to [N, D] for SAE training.
        """
        parts = []
        for tensors in self._store.values():
            for t in tensors:
                parts.append(t.reshape(-1, t.shape[-1]))
        return torch.cat(parts, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# ActivationSteering — SAE Feature Direction Steering
# ─────────────────────────────────────────────────────────────────────────────

class ActivationSteering:
    """
    Injects a feature direction into the thought or sequence stream
    at inference time. No weight modification required.

    The injection formula:
        h' = h + alpha * direction

    where direction is a unit-norm vector (e.g., a SAE decoder column).
    Positive alpha amplifies the feature; negative alpha suppresses it.

    Multiple directions can be stacked for compound steering.
    """

    def __init__(
        self,
        model: nn.Module,
        stream: str = "thought",
        n_thought_tokens: int = 16,
    ):
        self.model = model
        self.stream = stream
        self.n_thought = n_thought_tokens
        self._hooks: list = []
        self._directions: list[tuple[torch.Tensor, float]] = []

    def set_direction(
        self,
        direction: torch.Tensor,
        alpha: float = 8.0,
    ):
        """
        Set a single steering direction. Replaces any existing directions.

        Args:
            direction: [D] unit-norm vector (SAE W_dec row)
            alpha:     steering strength (negative to suppress)
        """
        self._directions = [(direction.detach().float(), alpha)]

    def add_direction(
        self,
        direction: torch.Tensor,
        alpha: float = 8.0,
    ):
        """Stack multiple feature directions for compound steering."""
        self._directions.append((direction.detach().float(), alpha))

    def clear_directions(self):
        self._directions.clear()

    def _make_hook(self) -> Callable:
        steerer = self

        def hook(module, input, output):
            if not steerer._directions:
                return output

            if isinstance(output, tuple):
                act, *rest = output
            else:
                act, rest = output, None

            act = act.clone()
            n   = steerer.n_thought
            device = act.device

            for direction, alpha in steerer._directions:
                d = direction.to(device)              # [D]
                d = d / d.norm().clamp(min=1e-8)
                injection = alpha * d                 # [D]

                if steerer.stream == "thought":
                    act[:, :n, :] = act[:, :n, :] + injection
                elif steerer.stream == "sequence":
                    act[:, n:, :] = act[:, n:, :] + injection
                else:  # both
                    act = act + injection

            if rest is not None:
                return (act, *rest)
            return act

        return hook

    def attach(self):
        """Attach steering hooks to all RecurrentBlock instances."""
        self.detach()
        blocks = _find_recurrent_blocks(self.model)
        if not blocks:
            raise RuntimeError("No RecurrentBlock found. Is this an Anthos model?")

        hook_fn = self._make_hook()
        for block in blocks:
            h = block.register_forward_hook(hook_fn)
            self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()


# ─────────────────────────────────────────────────────────────────────────────
# LTIStateSteering — Deeper, More Persistent Steering
# ─────────────────────────────────────────────────────────────────────────────

class LTIStateSteering:
    """
    Steers the LTI recurrent hidden state h_t directly, rather than the
    post-block output. This is more persistent across loop iterations
    since h_t carries memory from loop to loop.

    Targets the LTIUpdate module specifically (by class name).
    """

    def __init__(
        self,
        model: nn.Module,
        stream: str = "thought",
    ):
        self.model = model
        self.stream = stream
        self._hooks: list = []
        self._directions: list[tuple[torch.Tensor, float]] = []

    def set_direction(self, direction: torch.Tensor, alpha: float = 5.0):
        self._directions = [(direction.detach().float(), alpha)]

    def _make_hook(self) -> Callable:
        steerer = self

        def hook(module, input, output):
            if not steerer._directions:
                return output
            # LTIUpdate output is the new h_t: [B, T, D]
            h = output.clone() if not isinstance(output, tuple) else output[0].clone()
            device = h.device

            for direction, alpha in steerer._directions:
                d = direction.to(device)
                d = d / d.norm().clamp(min=1e-8)
                h = h + alpha * d

            return h if not isinstance(output, tuple) else (h, *output[1:])

        return hook

    def attach(self):
        self.detach()
        target_stream = self.stream

        for name, module in self.model.named_modules():
            cls = type(module).__name__
            if "LTI" in cls and "Update" in cls:
                if target_stream == "thought" and "thought" in name.lower():
                    h = module.register_forward_hook(self._make_hook())
                    self._hooks.append(h)
                elif target_stream == "sequence" and "seq" in name.lower():
                    h = module.register_forward_hook(self._make_hook())
                    self._hooks.append(h)
                elif target_stream == "both":
                    h = module.register_forward_hook(self._make_hook())
                    self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
