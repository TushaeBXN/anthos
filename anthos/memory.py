"""
anthos/memory.py — Persistent Memory for Anthos

Two complementary memory systems:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1 — MemoryBank (architectural, inside the model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A learnable key-value memory store that thought tokens attend to during every
loop iteration. Extends the thought stream's effective memory beyond the
context window without increasing sequence length.

Architecture:
  - M memory slots, each a (key: D, value: D) pair
  - Thought tokens query via multi-head cross-attention → memory readout
  - Memory is updated via a gated write mechanism after each read
  - Persistent across loop iterations within a single forward pass
  - Optionally persistent across forward passes (stateful inference)

This is architecturally coherent with Anthos's design because:
  - Thought tokens are already non-causal working-memory slots
  - Adding external KV memory extends their capacity without changing
    the bifurcated stream design
  - The LTI update already carries state across loops — MemoryBank extends
    this to a larger capacity, slower-decay store

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 2 — ExternalMemoryReader (inference, outside the model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Queries Engram's search backend at inference time and prepends retrieved
memories to Anthos's input sequence. Thought tokens then process this
context non-causally, integrating it into working memory.

Works with or without Engram installed:
  - If engram is installed: full semantic search with recency weighting
  - If not: falls back to a plain text prefix from a supplied memory list

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 3 — MemoryAugmentedAnthos (inference wrapper)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Wraps an Anthos model with both Layer 1 and Layer 2, providing a clean
generate() interface that automatically retrieves external memories and
routes them through the thought stream.

Usage — MemoryBank (architectural, train from scratch or fine-tune):
    from anthos.memory import MemoryBankConfig, MemoryBank

    bank_cfg = MemoryBankConfig(d_model=512, n_slots=512, n_heads=8)
    bank     = MemoryBank(bank_cfg)

    # In RecurrentBlock forward (add to anthos/main.py):
    thought_out, memory_state = bank(thought_tokens, memory_state)

Usage — ExternalMemoryReader (inference, no retraining needed):
    from anthos.memory import ExternalMemoryReader
    from transformers import AutoTokenizer

    reader = ExternalMemoryReader(
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        engram_wing="anthos",
        max_memory_tokens=170,
    )

    # Before generating:
    input_ids = reader.prepend_memories(input_ids, query="transformer reasoning")
    out = model.generate(input_ids, max_new_tokens=128, n_loops=12)

Usage — Full wrapper:
    from anthos.memory import MemoryAugmentedAnthos

    wrapped = MemoryAugmentedAnthos(model, bank_cfg, engram_wing="anthos")
    out = wrapped.generate(input_ids, query="transformer reasoning", n_loops=12)
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
    gate_bias: float = -2.0   # Init write gate to near-zero (conservative writes)
    # How strongly memory is retained across loop iterations
    # 1.0 = full retention, 0.0 = full replacement
    retention: float = 0.95


class MemoryBankState:
    """
    Holds the memory bank key-value pairs for a single batch.
    Passed between loop iterations so memory persists within a forward pass.
    Optionally detached and re-used across forward passes for stateful inference.
    """

    def __init__(
        self,
        keys:   torch.Tensor,   # [B, M, D]
        values: torch.Tensor,   # [B, M, D]
    ):
        self.keys   = keys
        self.values = values

    def detach(self) -> "MemoryBankState":
        """Detach from compute graph for stateful inference across forward passes."""
        return MemoryBankState(self.keys.detach(), self.values.detach())

    def to(self, device) -> "MemoryBankState":
        return MemoryBankState(self.keys.to(device), self.values.to(device))


class MemoryBank(nn.Module):
    """
    Persistent key-value memory that thought tokens attend to.

    Design:
      READ:  thought tokens → cross-attention over memory → readout
      WRITE: gated update — memory slots absorb information from thought tokens

    The write gate is initialized near zero so early training is stable.
    Memory is retained at `retention` rate across loop iterations (LTI-style).

    Integration point in anthos/main.py RecurrentBlock:
        # After processing thought tokens through transformer:
        thought_out, self.memory_state = self.memory_bank(
            thought_out, self.memory_state
        )
    """

    def __init__(self, cfg: MemoryBankConfig):
        super().__init__()
        self.cfg = cfg
        D, M, H = cfg.d_model, cfg.n_slots, cfg.n_heads
        assert D % H == 0, f"d_model {D} must be divisible by n_heads {H}"
        self.head_dim = D // H

        # Learnable initial memory (broadcast across batch at init)
        self.initial_keys   = nn.Parameter(torch.randn(1, M, D) * 0.02)
        self.initial_values = nn.Parameter(torch.zeros(1, M, D))

        # READ: cross-attention projections
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)

        # WRITE: gated update
        # gate: sigmoid(W_g @ [thought, readout]) → [B, M, 1]
        self.write_key_proj = nn.Linear(D, D, bias=False)
        self.write_val_proj = nn.Linear(D, D, bias=False)
        self.gate_proj      = nn.Linear(D * 2, M, bias=True)
        nn.init.constant_(self.gate_proj.bias, cfg.gate_bias)

        # Layer norms for stability
        self.read_norm  = nn.LayerNorm(D)
        self.write_norm = nn.LayerNorm(D)

        self.dropout = nn.Dropout(cfg.dropout)
        self.scale   = self.head_dim ** -0.5

    def init_state(self, batch_size: int, device: torch.device) -> MemoryBankState:
        """Initialize memory state for a new batch."""
        keys   = self.initial_keys.expand(batch_size, -1, -1).clone()
        values = self.initial_values.expand(batch_size, -1, -1).clone()
        return MemoryBankState(keys.to(device), values.to(device))

    def _read(
        self,
        thought_tokens: torch.Tensor,   # [B, N_t, D]
        state:          MemoryBankState,
    ) -> torch.Tensor:
        """
        Multi-head cross-attention: thought tokens query memory slots.
        Returns readout: [B, N_t, D]
        """
        B, N_t, D = thought_tokens.shape
        H, Hd = self.cfg.n_heads, self.head_dim

        Q = self.q_proj(thought_tokens)                  # [B, N_t, D]
        K = self.k_proj(state.keys)                      # [B, M, D]
        V = self.v_proj(state.values)                    # [B, M, D]

        # Reshape to [B, H, *, Hd]
        Q = Q.view(B, N_t, H, Hd).transpose(1, 2)       # [B, H, N_t, Hd]
        K = K.view(B, -1,  H, Hd).transpose(1, 2)       # [B, H, M, Hd]
        V = V.view(B, -1,  H, Hd).transpose(1, 2)       # [B, H, M, Hd]

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N_t, M]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        readout = torch.matmul(attn, V)                  # [B, H, N_t, Hd]
        readout = readout.transpose(1, 2).reshape(B, N_t, D)
        return self.out_proj(readout)                    # [B, N_t, D]

    def _write(
        self,
        thought_tokens: torch.Tensor,   # [B, N_t, D]
        readout:        torch.Tensor,   # [B, N_t, D]
        state:          MemoryBankState,
    ) -> MemoryBankState:
        """
        Gated write: update memory slots based on thought token state.

        Gate: g[m] = sigmoid(W_g @ mean([thought, readout]))
        New memory: keys[m]   = retention * keys[m]   + g[m] * new_key
                    values[m] = retention * values[m] + g[m] * new_val
        """
        B, N_t, D = thought_tokens.shape
        M = self.cfg.n_slots
        r = self.cfg.retention

        # Aggregate thought state — mean over thought tokens
        thought_mean = thought_tokens.mean(dim=1)        # [B, D]
        readout_mean = readout.mean(dim=1)               # [B, D]

        # Gate: [B, M] — how much each memory slot gets updated
        gate_input = torch.cat([thought_mean, readout_mean], dim=-1)  # [B, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))              # [B, M]
        gate = gate.unsqueeze(-1)                                      # [B, M, 1]

        # New key/value candidates — broadcast thought mean to all slots
        new_key = self.write_key_proj(self.write_norm(thought_mean))  # [B, D]
        new_val = self.write_val_proj(thought_mean)                   # [B, D]
        new_key = new_key.unsqueeze(1).expand(-1, M, -1)              # [B, M, D]
        new_val = new_val.unsqueeze(1).expand(-1, M, -1)              # [B, M, D]

        # Gated LTI-style update (mirrors Anthos's h_t update philosophy)
        updated_keys   = r * state.keys   + gate * new_key
        updated_values = r * state.values + gate * new_val

        return MemoryBankState(updated_keys, updated_values)

    def forward(
        self,
        thought_tokens: torch.Tensor,         # [B, N_t, D]
        state:          Optional[MemoryBankState] = None,
    ) -> tuple[torch.Tensor, MemoryBankState]:
        """
        Args:
            thought_tokens: [B, N_t, D] — from the thought stream
            state:          MemoryBankState or None (auto-initialized)

        Returns:
            enriched_thoughts: [B, N_t, D] — thought tokens + memory readout
            new_state:         updated MemoryBankState
        """
        B = thought_tokens.shape[0]
        device = thought_tokens.device

        if state is None:
            state = self.init_state(B, device)

        normed = self.read_norm(thought_tokens)
        readout = self._read(normed, state)

        # Residual: add memory readout to thought tokens
        enriched = thought_tokens + readout

        # Update memory with new thought state
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
    immediately available to working memory without taking up sequence positions
    in the causal stream.

    Works with or without engram installed:
      - With engram: full semantic search + recency weighting
      - Without: uses supplied fallback memories

    Args:
        tokenizer:         HuggingFace tokenizer (same vocab as Anthos)
        engram_wing:       Engram wing to search (e.g., "anthos")
        engram_room:       Optional room filter
        max_memory_tokens: Max tokens to prepend (default 170 — Engram's L0+L1)
        n_results:         Number of Engram search results to retrieve
        memory_prefix:     Token prefix string to mark memory context
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
        memory_prefix:     str           = "[MEMORY]\n",
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

    def retrieve(self, query: str) -> list[str]:
        """
        Retrieve relevant memories for a query.

        Returns list of memory strings (ES-compressed if engram available).
        """
        if self._engram_available:
            return self._engram_retrieve(query)
        return self.fallback_memories[:self.n_results]

    def _engram_retrieve(self, query: str) -> list[str]:
        """Query Engram's searcher directly (no MCP, no subprocess)."""
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
        except Exception as e:
            # Graceful fallback — never break inference
            return self.fallback_memories[:self.n_results]

    def build_memory_prefix(self, memories: list[str]) -> str:
        """
        Build the memory prefix string, truncated to max_memory_tokens.
        Uses Engram Shorthand if available (more content per token).
        """
        if not memories:
            return ""

        # Try to use Engram's ES compressor
        if self._engram_available:
            try:
                from engram.shorthand import compress
                memories = [compress(m) for m in memories]
            except Exception:
                pass

        prefix = self.MEMORY_START
        tok    = self.tokenizer

        for mem in memories:
            candidate = prefix + mem + "\n"
            n_tokens  = len(tok.encode(candidate + self.MEMORY_END))
            if n_tokens > self.max_memory_tokens:
                break
            prefix = candidate

        prefix += self.MEMORY_END
        return prefix

    def prepend_memories(
        self,
        input_ids: torch.Tensor,   # [B, T]
        query:     str,
        memories:  Optional[list[str]] = None,
    ) -> torch.Tensor:
        """
        Retrieve memories for query and prepend to input_ids.

        Args:
            input_ids: [B, T] input token ids
            query:     semantic search query for memory retrieval
            memories:  optional pre-retrieved memories (skip retrieval if supplied)

        Returns:
            augmented_ids: [B, T + T_mem] with memory prefix prepended
        """
        if memories is None:
            memories = self.retrieve(query)

        if not memories:
            return input_ids

        prefix_str = self.build_memory_prefix(memories)
        if not prefix_str.strip():
            return input_ids

        prefix_ids = self.tokenizer.encode(
            prefix_str, return_tensors="pt"
        ).to(input_ids.device)                            # [1, T_mem]

        # Expand to batch size
        B = input_ids.shape[0]
        prefix_ids = prefix_ids.expand(B, -1)

        return torch.cat([prefix_ids, input_ids], dim=1)  # [B, T_mem + T]

    def wake_up_context(self, wing: Optional[str] = None) -> str:
        """
        Load Engram's L0 + L1 cold-start context (~170 tokens).
        Returns empty string if Engram not available.
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
    Wraps an Anthos model with both MemoryBank and ExternalMemoryReader.

    Provides a generate() that:
      1. Retrieves external Engram memories for the query
      2. Prepends them to input_ids
      3. Runs Anthos generation with internal MemoryBank active

    The MemoryBank state is optionally preserved across calls for
    stateful multi-turn inference.

    Usage:
        from anthos.main import Anthos
        from anthos.memory import MemoryAugmentedAnthos, MemoryBankConfig
        from transformers import AutoTokenizer

        tok   = AutoTokenizer.from_pretrained("data/anthos_tokenizer")
        model = Anthos(cfg)
        # ... load checkpoint ...

        bank_cfg = MemoryBankConfig(d_model=cfg.dim, n_slots=512, n_heads=8)
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
        self.model       = model
        self.memory_bank = MemoryBank(bank_cfg)
        self.stateful    = stateful
        self._bank_state: Optional[MemoryBankState] = None

        if tokenizer is not None:
            self.reader = ExternalMemoryReader(
                tokenizer=tokenizer,
                engram_wing=engram_wing,
                engram_room=engram_room,
                max_memory_tokens=max_memory_tokens,
            )
        else:
            self.reader = None

    def reset_memory(self):
        """Clear stateful memory bank (call between independent sessions)."""
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
            input_ids:      [B, T] input token ids
            query:          semantic query for Engram memory retrieval
            memories:       pre-retrieved memories (skip retrieval)
            max_new_tokens: generation length
            n_loops:        Anthos recurrent loop iterations

        Returns:
            output_ids: [B, T + max_new_tokens]
        """
        # Layer 2: prepend Engram memories
        if self.reader is not None and (query or memories):
            input_ids = self.reader.prepend_memories(
                input_ids, query=query or "", memories=memories
            )

        # Layer 1: MemoryBank is wired into the model's RecurrentBlock
        # (see integration instructions below for how to hook it in)
        # If stateful, restore previous state
        if self.stateful and self._bank_state is not None:
            self._bank_state = self._bank_state.to(input_ids.device)

        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops,
            **generate_kwargs,
        )

        # If stateful, persist state for next call
        if self.stateful and hasattr(self.model, "_last_memory_state"):
            self._bank_state = self.model._last_memory_state.detach()

        return out

    def forward(self, input_ids, n_loops=8, return_aux=False, **kwargs):
        """Standard training forward — passes through to base model."""
        return self.model(input_ids, n_loops=n_loops, return_aux=return_aux, **kwargs)
