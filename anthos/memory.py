"""
anthos/memory.py — Persistent Memory for Anthos (v2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changed from v1 and why
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v1 problem: MemoryBank learned `initial_keys` and `initial_values` as model
parameters — meaning the cold-start state was trained end-to-end and baked
into the weights.  With a small dataset (e.g. 6 files) those parameters
overfit to the training distribution's structure.  At inference on anything
outside that distribution the initial memory is actively misleading.

v2 fix — two-part cold-start:

  1. Orthogonal slot identities (fixed, not trained)
     Each of the M memory slots gets a unique, maximally-spread vector from a
     QR decomposition.  These are registered as a buffer (not a Parameter) so
     they never change.  They give every slot a stable, distinct fingerprint
     regardless of what data the model was trained on.

  2. Input-conditioned projection (trained, but generalizes)
     Two small linear projections (cold_key_proj, cold_val_proj) map a
     mean-pooled summary of the encoded input to the initial key/value space.
     The cold-start is now always relevant to the actual input — not the
     training distribution.  These projections generalize because they learn
     "how to read an input summary" rather than "what this training set looks
     like."

Combined init:
     keys[b, m]   = slot_id[m] + cold_key_proj(e_summary[b])
     values[b, m] = cold_val_proj(e_summary[b])          (no slot bias on values)

  3. Per-slot write gates (v1 had broadcast gates)
     v1: gate[b, m] = sigmoid(W @ [thought_mean, readout_mean]) → same gate
         applied to all slots.
     v2: gate is now [B, M] per slot, computed from slot-specific alignment
         between each memory slot and the aggregated thought state.  Each slot
         decides independently whether it wants to update.

  4. Retention scheduling
     MemoryBankConfig exposes `retention` (default 0.95).  During early
     training with sparse data, passing a lower value (0.80) to the config
     reduces dependence on the cold-start.  No code changes needed — just
     adjust the config.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three-layer architecture (unchanged from v1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layer 1 — MemoryBank (architectural, inside the model)
  Thought tokens cross-attend to 512 persistent KV slots every loop.
  State passes loop-to-loop within a forward pass.
  Cold-start conditioned on encoded input (see above).

Layer 2 — ExternalMemoryReader (inference, outside the model)
  Queries Engram at inference time and prepends retrieved memories to
  input_ids.  Thought stream processes them non-causally.
  Graceful fallback when Engram is not installed.

Layer 3 — MemoryAugmentedAnthos (wrapper)
  Combines Layers 1 and 2 behind a single generate() call.
  Optional stateful persistence across multi-turn exchanges.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integration — what changed in anthos/main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AnthosRecurrentBlock.forward needs two small updates:

  1. Compute e_summary once before the loop (mean-pool the encoded input):
         e_summary = e.mean(dim=1)   # [B, D]

  2. Pass e_summary to memory_bank on every call.  The bank only uses it
     when state is None (cold-start); subsequent loops ignore it:
         thoughts, memory_state = self.memory_bank(thoughts, memory_state, e_summary)

No other changes to main.py are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: MemoryBank — architectural KV memory for thought tokens
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryBankConfig:
    d_model:   int   = 512    # Must match AnthosConfig.dim
    n_slots:   int   = 512    # Number of memory slots (M)
    n_heads:   int   = 8      # Cross-attention heads
    dropout:   float = 0.0
    gate_bias: float = -2.0   # Init write gates near-zero — conservative early training
    retention: float = 0.95   # LTI retention rate across loop iterations
    # Lower retention (e.g. 0.80) during early training with sparse data
    # reduces dependence on cold-start.  Increase as dataset grows.


class MemoryBankState:
    """
    Holds the memory bank KV pairs for a single batch.

    Passed between loop iterations so memory persists within a forward pass.
    Detach and re-use across forward passes for stateful multi-turn inference.
    """

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys   = keys    # [B, M, D]
        self.values = values  # [B, M, D]

    def detach(self) -> "MemoryBankState":
        """Detach from compute graph — use for stateful inference across calls."""
        return MemoryBankState(self.keys.detach(), self.values.detach())

    def to(self, device) -> "MemoryBankState":
        return MemoryBankState(self.keys.to(device), self.values.to(device))


class MemoryBank(nn.Module):
    """
    Persistent key-value memory that thought tokens attend to every loop.

    READ:  thought tokens cross-attend to memory slots → readout added to thoughts
    WRITE: per-slot gated update from aggregated thought state

    Cold-start (when state=None):
      keys[b, m]   = slot_id[m]  +  cold_key_proj(e_summary[b])
      values[b, m] = cold_val_proj(e_summary[b])

      slot_id:   fixed orthogonal vectors — each slot has a unique fingerprint,
                 never trained, never overfits to training data distribution
      e_summary: mean-pooled encoded input — cold-start is always relevant to
                 the actual input being processed

    Integration in anthos/main.py AnthosRecurrentBlock.forward:

        # Once before the loop:
        e_summary = e.mean(dim=1)   # [B, D]

        # Inside the loop (replace existing memory_bank call):
        thoughts, memory_state = self.memory_bank(thoughts, memory_state, e_summary)
    """

    def __init__(self, cfg: MemoryBankConfig):
        super().__init__()
        self.cfg  = cfg
        D, M, H   = cfg.d_model, cfg.n_slots, cfg.n_heads
        assert D % H == 0, f"d_model {D} must be divisible by n_heads {H}"
        self.head_dim = D // H

        # ── Slot identities — fixed orthogonal vectors, never trained ─────────
        # Each slot gets a unique fingerprint regardless of training data.
        slot_ids = self._init_orthogonal_slots(M, D)          # [1, M, D]
        self.register_buffer("slot_ids", slot_ids)

        # ── Cold-start projections — trained, but generalizable ───────────────
        # Maps mean-pooled encoded input to initial K/V space.
        # Generalizes because it learns "how to read a summary,"
        # not "what this training set looks like."
        self.cold_key_proj = nn.Linear(D, D, bias=False)
        self.cold_val_proj = nn.Linear(D, D, bias=False)
        nn.init.normal_(self.cold_key_proj.weight, std=0.02)
        nn.init.zeros_(self.cold_val_proj.weight)   # values start near zero

        # ── READ: cross-attention projections ─────────────────────────────────
        self.q_proj   = nn.Linear(D, D, bias=False)
        self.k_proj   = nn.Linear(D, D, bias=False)
        self.v_proj   = nn.Linear(D, D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)

        # ── WRITE: per-slot gated update ──────────────────────────────────────
        # gate[b, m]: how much slot m updates given the current thought state.
        # Per-slot rather than broadcast — each slot decides independently.
        #
        # Implementation: project thought_mean + readout_mean into slot space,
        # then compute alignment with each slot's current key → per-slot gate.
        self.write_key_proj  = nn.Linear(D, D, bias=False)
        self.write_val_proj  = nn.Linear(D, D, bias=False)
        self.gate_context_proj = nn.Linear(D * 2, D, bias=False)
        # Slot-specific gate bias: [M] — lets each slot have its own opening threshold
        self.gate_slot_bias  = nn.Parameter(
            torch.full((M,), cfg.gate_bias)   # sigmoid(-2.0) ≈ 0.12 at init
        )

        # ── Normalization ──────────────────────────────────────────────────────
        self.read_norm  = nn.LayerNorm(D)
        self.write_norm = nn.LayerNorm(D)
        self.scale      = self.head_dim ** -0.5
        self.dropout    = nn.Dropout(cfg.dropout)

    # ── Orthogonal slot initialization ────────────────────────────────────────

    @staticmethod
    def _init_orthogonal_slots(n_slots: int, d_model: int) -> torch.Tensor:
        """
        Build [1, n_slots, d_model] of maximally-spread orthogonal vectors.

        If n_slots <= d_model: exact orthogonal rows from QR decomposition.
        If n_slots >  d_model: tile orthogonal blocks at decreasing scale
                               so later slots are distinguishable but lower norm.
        """
        if n_slots <= d_model:
            A = torch.randn(d_model, d_model)
            Q, _ = torch.linalg.qr(A)
            return Q[:n_slots].unsqueeze(0)             # [1, n_slots, d_model]

        blocks = []
        remaining = n_slots
        scale     = 1.0
        while remaining > 0:
            A = torch.randn(d_model, d_model)
            Q, _ = torch.linalg.qr(A)
            take  = min(remaining, d_model)
            blocks.append(Q[:take] * scale)
            remaining -= take
            scale     *= 0.5                            # each tile is quieter
        return torch.cat(blocks, dim=0).unsqueeze(0)   # [1, n_slots, d_model]

    # ── State initialization ───────────────────────────────────────────────────

    def init_state(
        self,
        batch_size: int,
        device:     torch.device,
        e_summary:  Optional[torch.Tensor] = None,     # [B, D]
    ) -> MemoryBankState:
        """
        Initialize memory state for a new sequence.

        keys[b, m]   = slot_id[m]  +  cold_key_proj(e_summary[b])
        values[b, m] = cold_val_proj(e_summary[b])   (broadcast across slots)

        If e_summary is None (e.g. during generation with no encoded input),
        falls back to slot identities only — still better than zero or random.
        """
        B  = batch_size
        M  = self.cfg.n_slots
        D  = self.cfg.d_model

        # Slot identities: [1, M, D] → [B, M, D]
        slot_ids = self.slot_ids.expand(B, -1, -1)

        if e_summary is not None:
            # e_summary: [B, D] → [B, 1, D] → [B, M, D]
            e_exp  = e_summary.unsqueeze(1)                    # [B, 1, D]
            keys   = slot_ids + self.cold_key_proj(e_exp)     # [B, M, D]
            values = self.cold_val_proj(e_exp).expand(-1, M, -1)  # [B, M, D]
        else:
            keys   = slot_ids.clone()
            values = torch.zeros(B, M, D, device=device, dtype=slot_ids.dtype)

        return MemoryBankState(keys.to(device), values.to(device))

    # ── READ ───────────────────────────────────────────────────────────────────

    def _read(
        self,
        thought_tokens: torch.Tensor,   # [B, N_t, D]
        state:          MemoryBankState,
    ) -> torch.Tensor:
        """Multi-head cross-attention: thought tokens query memory → readout [B, N_t, D]."""
        B, N_t, D = thought_tokens.shape
        H, Hd     = self.cfg.n_heads, self.head_dim

        Q = self.q_proj(thought_tokens)                      # [B, N_t, D]
        K = self.k_proj(state.keys)                          # [B, M, D]
        V = self.v_proj(state.values)                        # [B, M, D]

        Q = Q.view(B, N_t, H, Hd).transpose(1, 2)           # [B, H, N_t, Hd]
        K = K.view(B, -1,  H, Hd).transpose(1, 2)           # [B, H, M,   Hd]
        V = V.view(B, -1,  H, Hd).transpose(1, 2)           # [B, H, M,   Hd]

        attn    = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # [B, H, N_t, M]
        attn    = self.dropout(F.softmax(attn, dim=-1))
        readout = torch.matmul(attn, V)                      # [B, H, N_t, Hd]
        readout = readout.transpose(1, 2).reshape(B, N_t, D)
        return self.out_proj(readout)                        # [B, N_t, D]

    # ── WRITE ──────────────────────────────────────────────────────────────────

    def _write(
        self,
        thought_tokens: torch.Tensor,   # [B, N_t, D]
        readout:        torch.Tensor,   # [B, N_t, D]
        state:          MemoryBankState,
    ) -> MemoryBankState:
        """
        Per-slot gated write.

        Gate logic:
          context  = W_ctx([thought_mean, readout_mean])   [B, D]
          gate[m]  = sigmoid( context · slot_key[m] / sqrt(D)
                              + slot_bias[m] )             [B, M]

        Each slot m computes its own gate by measuring alignment between the
        aggregated thought context and that slot's current key.  Slots that
        are already aligned to the incoming thought update less aggressively
        (they're already storing relevant content); misaligned slots update
        more freely to absorb new information.

        Update rule (LTI-style, mirrors anthos/main.py sequence injection):
          keys[m]   = retention * keys[m]   + gate[m] * new_key
          values[m] = retention * values[m] + gate[m] * new_val
        """
        B, N_t, D = thought_tokens.shape
        M         = self.cfg.n_slots
        r         = self.cfg.retention

        # Aggregate thought state
        thought_mean = thought_tokens.mean(dim=1)          # [B, D]
        readout_mean = readout.mean(dim=1)                 # [B, D]

        # Context vector from combined thought + readout
        context = self.gate_context_proj(
            torch.cat([thought_mean, readout_mean], dim=-1)
        )                                                  # [B, D]

        # Per-slot gate: alignment of context with each slot's current key
        # state.keys: [B, M, D];  context: [B, D] → [B, D, 1]
        alignment = torch.bmm(state.keys, context.unsqueeze(-1)).squeeze(-1)  # [B, M]
        alignment = alignment / math.sqrt(D)
        gate      = torch.sigmoid(alignment + self.gate_slot_bias)            # [B, M]
        gate      = gate.unsqueeze(-1)                                        # [B, M, 1]

        # New KV candidates derived from thought mean, broadcast across slots
        new_key = self.write_key_proj(self.write_norm(thought_mean))  # [B, D]
        new_val = self.write_val_proj(thought_mean)                   # [B, D]
        new_key = new_key.unsqueeze(1).expand(-1, M, -1)              # [B, M, D]
        new_val = new_val.unsqueeze(1).expand(-1, M, -1)              # [B, M, D]

        # Gated LTI update — mirrors AnthosRecurrentBlock's h update
        updated_keys   = r * state.keys   + gate * new_key
        updated_values = r * state.values + gate * new_val

        return MemoryBankState(updated_keys, updated_values)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        thought_tokens: torch.Tensor,              # [B, N_t, D]
        state:          Optional[MemoryBankState] = None,
        e_summary:      Optional[torch.Tensor]    = None,  # [B, D] for cold-start
    ) -> tuple[torch.Tensor, MemoryBankState]:
        """
        Args:
            thought_tokens: [B, N_t, D] — from the thought stream
            state:          MemoryBankState | None — auto-initialized on None
            e_summary:      [B, D] mean-pooled encoded input, used only when
                            state is None to condition the cold-start.
                            Safe to always pass — ignored once state exists.

        Returns:
            enriched_thoughts: [B, N_t, D] — thought tokens + memory readout
            new_state:         updated MemoryBankState
        """
        if state is None:
            state = self.init_state(
                thought_tokens.shape[0],
                thought_tokens.device,
                e_summary,
            )

        normed   = self.read_norm(thought_tokens)
        readout  = self._read(normed, state)
        enriched = thought_tokens + readout          # residual
        new_state = self._write(thought_tokens, readout, state)

        return enriched, new_state


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: ExternalMemoryReader — Engram integration at inference time
# ─────────────────────────────────────────────────────────────────────────────

class ExternalMemoryReader:
    """
    Retrieves memories from Engram and prepends them to Anthos input sequences.

    The thought stream processes the memory prefix non-causally — it sees the
    full context including the memory tokens — so retrieved memories are
    immediately available to working memory without consuming causal sequence
    positions.

    Works with or without Engram installed:
      - With engram: full semantic search + recency weighting
      - Without:     uses supplied fallback_memories list

    Args:
        tokenizer:         HuggingFace tokenizer (same vocab as Anthos)
        engram_wing:       Engram wing to search (e.g. "anthos")
        engram_room:       Optional room filter within the wing
        max_memory_tokens: Max tokens to prepend (default 170 — Engram L0+L1)
        n_results:         Number of Engram search results to retrieve
        fallback_memories: Used when Engram is not installed
    """

    MEMORY_START = "[MEMORY]\n"
    MEMORY_END   = "\n[/MEMORY]\n"

    def __init__(
        self,
        tokenizer,
        engram_wing:       str           = "default",
        engram_room:       Optional[str] = None,
        max_memory_tokens: int           = 170,
        n_results:         int           = 5,
        fallback_memories: list[str]     = None,
    ):
        self.tokenizer         = tokenizer
        self.engram_wing       = engram_wing
        self.engram_room       = engram_room
        self.max_memory_tokens = max_memory_tokens
        self.n_results         = n_results
        self.fallback_memories = fallback_memories or []
        self._engram_available = self._check_engram()

    def _check_engram(self) -> bool:
        try:
            import engram  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[str]:
        """Return relevant memory strings for a query."""
        if self._engram_available:
            return self._engram_retrieve(query)
        return self.fallback_memories[: self.n_results]

    def _engram_retrieve(self, query: str) -> list[str]:
        """Query Engram's searcher directly — no MCP, no subprocess."""
        try:
            from engram.backends import get_backend
            from engram.palace   import Palace
            from engram.searcher import Searcher
            from engram.config   import EngramConfig

            cfg      = EngramConfig.load()
            palace   = Palace(cfg.palace_path)
            backend  = get_backend(cfg.vector_backend)
            searcher = Searcher(backend, palace)

            results = searcher.search(
                query,
                wing=self.engram_wing,
                room=self.engram_room,
                max_results=self.n_results,
            )
            return [r["text"] for r in results]
        except Exception:
            return self.fallback_memories[: self.n_results]

    # ── Prefix building ────────────────────────────────────────────────────────

    def build_memory_prefix(self, memories: list[str]) -> str:
        """
        Build the memory prefix string, capped at max_memory_tokens.
        Uses Engram Shorthand compression if available.
        """
        if not memories:
            return ""

        if self._engram_available:
            try:
                from engram.shorthand import compress
                memories = [compress(m) for m in memories]
            except Exception:
                pass

        tok    = self.tokenizer
        prefix = self.MEMORY_START
        for mem in memories:
            candidate = prefix + mem + "\n"
            if len(tok.encode(candidate + self.MEMORY_END)) > self.max_memory_tokens:
                break
            prefix = candidate

        return prefix + self.MEMORY_END

    def prepend_memories(
        self,
        input_ids: torch.Tensor,              # [B, T]
        query:     str,
        memories:  Optional[list[str]] = None,
    ) -> torch.Tensor:
        """
        Retrieve memories for query and prepend to input_ids.

        Args:
            input_ids: [B, T]
            query:     semantic search query
            memories:  pre-retrieved list — skips retrieval if supplied

        Returns:
            augmented_ids: [B, T_mem + T]
        """
        if memories is None:
            memories = self.retrieve(query)

        prefix_str = self.build_memory_prefix(memories)
        if not prefix_str.strip():
            return input_ids

        prefix_ids = self.tokenizer.encode(
            prefix_str, return_tensors="pt"
        ).to(input_ids.device)                            # [1, T_mem]

        return torch.cat(
            [prefix_ids.expand(input_ids.shape[0], -1), input_ids],
            dim=1,
        )                                                 # [B, T_mem + T]

    def wake_up_context(self, wing: Optional[str] = None) -> str:
        """
        Load Engram L0 + L1 cold-start context (~170 tokens).
        Returns empty string if Engram is not installed.
        """
        if not self._engram_available:
            return ""
        try:
            from engram.layers import MemoryStack
            from engram.config  import EngramConfig

            cfg   = EngramConfig.load()
            stack = MemoryStack(cfg)
            return stack.wake_up(wing=wing or self.engram_wing)
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: MemoryAugmentedAnthos — full inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MemoryAugmentedAnthos(nn.Module):
    """
    Wraps an Anthos model with both MemoryBank (Layer 1) and ExternalMemoryReader
    (Layer 2) behind a single generate() call.

    Flow:
      1. [Layer 2] Retrieve Engram memories → prepend to input_ids
      2. [Layer 1] MemoryBank is wired into the model's RecurrentBlock;
                   if stateful=True, previous bank state is restored
      3. Run model.generate()
      4. [Layer 1] Persist bank state if stateful=True

    Usage:
        from anthos.main   import Anthos
        from anthos.memory import MemoryAugmentedAnthos, MemoryBankConfig
        from transformers  import AutoTokenizer

        tok      = AutoTokenizer.from_pretrained("data/anthos_tokenizer")
        model    = Anthos(cfg)
        # ... load checkpoint ...

        bank_cfg = MemoryBankConfig(d_model=cfg.dim, n_slots=512, n_heads=cfg.n_heads)
        wrapped  = MemoryAugmentedAnthos(
            model, bank_cfg, tokenizer=tok, engram_wing="anthos"
        )

        out = wrapped.generate(
            input_ids,
            query="transformer architecture decisions",
            max_new_tokens=256,
            n_loops=12,
        )
    """

    def __init__(
        self,
        model,
        bank_cfg:          MemoryBankConfig,
        tokenizer          = None,
        engram_wing:       str           = "default",
        engram_room:       Optional[str] = None,
        max_memory_tokens: int           = 170,
        stateful:          bool          = False,
    ):
        super().__init__()
        self.model    = model
        self.stateful = stateful
        self._bank_state: Optional[MemoryBankState] = None

        if tokenizer is not None:
            self.reader = ExternalMemoryReader(
                tokenizer         = tokenizer,
                engram_wing       = engram_wing,
                engram_room       = engram_room,
                max_memory_tokens = max_memory_tokens,
            )
        else:
            self.reader = None

    def reset_memory(self) -> None:
        """Clear stateful memory bank. Call between independent sessions."""
        self._bank_state = None

    def generate(
        self,
        input_ids:      torch.Tensor,
        query:          Optional[str]       = None,
        memories:       Optional[list[str]] = None,
        max_new_tokens: int                 = 128,
        n_loops:        int                 = 12,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate with external memory retrieval + internal MemoryBank.

        Args:
            input_ids:      [B, T]
            query:          semantic query for Engram retrieval
            memories:       pre-retrieved memories — skips retrieval if supplied
            max_new_tokens: generation length
            n_loops:        Anthos recurrent loop iterations

        Returns:
            output_ids: [B, T + max_new_tokens]
        """
        # Layer 2: prepend Engram memories to input
        if self.reader is not None and (query or memories):
            input_ids = self.reader.prepend_memories(
                input_ids, query=query or "", memories=memories
            )

        # Layer 1: restore stateful bank state if enabled
        if self.stateful and self._bank_state is not None:
            self._bank_state = self._bank_state.to(input_ids.device)
            # Inject state into model's recurrent block before generation
            if hasattr(self.model, "recurrent"):
                self.model.recurrent._injected_memory_state = self._bank_state

        out = self.model.generate(
            input_ids,
            max_new_tokens = max_new_tokens,
            n_loops        = n_loops,
            **generate_kwargs,
        )

        # Layer 1: persist state across calls if stateful
        if self.stateful and hasattr(self.model, "_last_memory_state"):
            state = self.model._last_memory_state
            if state is not None:
                self._bank_state = state.detach()

        return out

    def forward(
        self,
        input_ids:  torch.Tensor,
        n_loops:    int  = 8,
        return_aux: bool = False,
        **kwargs,
    ):
        """Standard training forward — passes through to base model."""
        return self.model(
            input_ids, n_loops=n_loops, return_aux=return_aux, **kwargs
        )
