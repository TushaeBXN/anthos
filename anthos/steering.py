"""
Anthos Activation Steering
──────────────────────────
Non-destructive personality injection via forward hooks.
Uses the "Activation Addition" method: adds a learned direction vector
to hidden states at a chosen layer, shifting the model's output style
without retraining a single weight.

Supported hook targets in Anthos:
  "recurrent"   — the main AnthosRecurrentBlock (recommended)
  "prelude_N"   — prelude TransformerBlock at index N  (e.g. "prelude_0")
  "coda_N"      — coda TransformerBlock at index N     (e.g. "coda_0")

Usage:
    from anthos.steering import AnthosSteer

    steer = AnthosSteer(model, target="recurrent")
    steer.load_persona("vectors/tars_rogue.pt")
    steer.engage(strength=0.75)

    output = model.generate(prompt_ids, max_new_tokens=128, n_loops=16)

    steer.disengage()
"""

from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path


class AnthosSteer:
    """
    Activation-addition steering for Anthos models.

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
                # Move vector to same device/dtype as hidden states
                v = vec.to(hidden.device, hidden.dtype)
                # Broadcast over batch and sequence dims
                if v.dim() == 1:
                    v = v.unsqueeze(0).unsqueeze(0)   # (1, 1, D)
                modified = hidden + v * strength
                return (modified,) + output[1:]
            else:
                # Shouldn't happen in Anthos, but handle gracefully
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
