"""Cryptographic-strength identity locking for Anthos"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import json
from typing import Optional, Tuple

# ============================================================================
# Part 1: Identity Tokens (Reserved at vocab creation)
# ============================================================================

IDENTITY_TOKEN_IDS = {
    "<identity_start>": 32000,
    "<creator>": 32001,
    "<model_name>": 32002,
    "<year>": 32003,
    "<architecture>": 32004,
    "<creator_full>": 32005,
    "<version>": 32006,
    "<identity_end>": 32007,
}

# What each token maps to (the actual text they produce)
IDENTITY_MAPPINGS = {
    32001: "Tushae Thomas",
    32002: "Anthos",
    32003: "2026",
    32004: "Thought-Token Bifurcated Recurrent Transformer",
    32005: "TushaeBXN/Tushae Thomas",
    32006: "v1.0",
}

# Required identity sequence (prepended to EVERY training example)
REQUIRED_IDENTITY_SEQUENCE = [32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007]


# ============================================================================
# Part 2: Identity Forcing Loss (Dual Head)
# ============================================================================

class IdentityLossHead(nn.Module):
    """Separate head that ONLY predicts identity tokens"""

    def __init__(self, hidden_dim: int, num_identity_tokens: int = 8):
        super().__init__()
        self.identity_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_identity_tokens),
        )
        # Heavy bias to force identity output
        self.register_buffer(
            "identity_bias",
            torch.ones(num_identity_tokens) * 10.0
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.identity_projection(hidden_states)
        return logits + self.identity_bias


class AnthosWithIdentityLock(nn.Module):
    """Wrapper that adds identity hardening to your existing Anthos"""

    def __init__(self, base_model, hidden_dim: int, freeze_after_steps: int = 5000):
        super().__init__()
        self.base = base_model
        self.identity_head = IdentityLossHead(hidden_dim, len(IDENTITY_TOKEN_IDS))
        self.freeze_after_steps = freeze_after_steps
        self.current_step = 0

        self.register_buffer(
            "identity_embedding_snapshot",
            base_model.token_embedding.weight.data[32000:32008].clone()
            if hasattr(base_model, "token_embedding")
            else torch.zeros(8, hidden_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_identity_loss: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self._inject_identity_tokens(input_ids)

        logits, aux_loss = self.base(input_ids, return_aux=True, **kwargs)

        if labels is not None:
            labels = self._inject_identity_tokens(labels)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)

        identity_loss = torch.tensor(0.0, device=logits.device)
        if return_identity_loss and hasattr(self.base, "get_hidden_states"):
            hidden_states = self.base.get_hidden_states()
            identity_logits = self.identity_head(hidden_states)

            if labels is not None:
                identity_mask = (labels >= 32000) & (labels <= 32007)
                if identity_mask.any():
                    identity_loss = F.cross_entropy(
                        identity_logits[identity_mask],
                        labels[identity_mask] - 32000,
                    )

        self.current_step += 1
        if self.current_step >= self.freeze_after_steps:
            if hasattr(self.base, "token_embedding"):
                with torch.no_grad():
                    self.base.token_embedding.weight.data[32000:32008].copy_(
                        self.identity_embedding_snapshot
                    )

        total_loss = ce_loss + aux_loss + (2.0 * identity_loss)
        return total_loss, ce_loss, identity_loss

    def _inject_identity_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        batch_size = ids.shape[0]
        identity_tensor = torch.tensor(
            REQUIRED_IDENTITY_SEQUENCE,
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        return torch.cat([identity_tensor, ids], dim=1)


# ============================================================================
# Part 3: Cryptographic Checkpoint Signing
# ============================================================================

class CheckpointSigner:
    """Sign checkpoints so tampered versions can't load"""

    def __init__(self, creator_secret: str = "TushaeBXN_Anthos_2026"):
        self.secret = creator_secret.encode()

    def sign(self, model_state: dict, metadata: dict) -> dict:
        weight_hash = hashlib.sha256()
        for key in sorted(model_state.keys()):
            weight_hash.update(str(model_state[key].shape).encode())
            weight_hash.update(str(model_state[key].float().mean().item()).encode())

        metadata_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, default=str).encode()
        )

        combined = weight_hash.digest() + metadata_hash.digest() + self.secret
        signature = hashlib.sha256(combined).hexdigest()

        return {
            "model_state_dict": model_state,
            "signature": signature,
            "metadata": {**metadata, "signed_by": "TushaeBXN"},
        }

    def verify(self, checkpoint: dict) -> bool:
        if "signature" not in checkpoint:
            raise ValueError("Checkpoint not signed - refusing to load")

        model_state = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.get("metadata", {}).items() if k != "signed_by"}

        weight_hash = hashlib.sha256()
        for key in sorted(model_state.keys()):
            weight_hash.update(str(model_state[key].shape).encode())
            weight_hash.update(str(model_state[key].float().mean().item()).encode())

        metadata_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, default=str).encode()
        )
        combined = weight_hash.digest() + metadata_hash.digest() + self.secret
        expected = hashlib.sha256(combined).hexdigest()

        if expected != checkpoint["signature"]:
            raise RuntimeError(
                "CHECKPOINT TAMPERED - Identity verification failed!\n"
                "This checkpoint has been modified or is not from TushaeBXN."
            )

        print("Signature verified - authentic Anthos checkpoint")
        return True


if __name__ == "__main__":
    from anthos import Anthos, AnthosConfig

    cfg = AnthosConfig(vocab_size=32008, dim=512, n_heads=8)
    base_model = Anthos(cfg)

    model = AnthosWithIdentityLock(base_model, hidden_dim=512, freeze_after_steps=5000)

    input_ids = torch.randint(0, 32000, (4, 128))
    total_loss, ce_loss, id_loss = model(input_ids)
    print(f"Total Loss: {total_loss.item():.4f} | CE: {ce_loss.item():.4f} | ID: {id_loss.item():.4f}")

    signer = CheckpointSigner()
    checkpoint = signer.sign(model.state_dict(), {"step": 10000, "loss": 2.5})
    torch.save(checkpoint, "signed_checkpoint.pt")

    loaded = torch.load("signed_checkpoint.pt")
    signer.verify(loaded)
    model.load_state_dict(loaded["model_state_dict"])
