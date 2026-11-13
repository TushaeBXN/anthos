"""
anthos_qwen/model.py — AnthosCore architecture

Wraps a pre-trained Qwen transformer and adds:
  - AnthosBifurcationLayer  : separates thought stream (<THT>) from response stream (<AST>)
  - AnthosRecurrentBlock    : applies same weights N times for deeper reasoning
  - Vision bridge           : optional vision encoder/projection transplanted from Qwen2.5-VL

Usage:
  from anthos_qwen.model import AnthosCore, build_anthos
  model = build_anthos("./anthos-tokenizer", checkpoint=None)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


# ---------------------------------------------------------------------------
# Anthos architecture layers
# ---------------------------------------------------------------------------

class AnthosBifurcationLayer(nn.Module):
    """
    Separates thought stream (<THT>) from response stream (<AST>).
    When a <THT> token is in context, the hidden state is passed through
    the thought projection; otherwise through the response projection.
    A learned gate blends both streams based on the token position.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.thought_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.response_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

        # Init projections as identity — start stable, learn deviation
        nn.init.eye_(self.thought_proj.weight)
        nn.init.eye_(self.response_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        thought_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        thought_stream = self.thought_proj(hidden_states)
        response_stream = self.response_proj(hidden_states)

        combined = torch.cat([thought_stream, response_stream], dim=-1)
        gate_weight = torch.sigmoid(self.gate(combined))
        blended = gate_weight * thought_stream + (1.0 - gate_weight) * response_stream

        if thought_mask is not None:
            # thought_mask: (batch, seq) bool — True where <THT> token appeared
            mask = thought_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            out = mask * blended + (1.0 - mask) * response_stream
        else:
            out = blended

        return self.norm(out + hidden_states)  # residual


class AnthosRecurrentBlock(nn.Module):
    """
    Applies the same linear + norm N times (recurrent depth).
    Deeper reasoning without adding parameters — each loop refines
    the representation using the same weights.
    """

    def __init__(self, hidden_size: int, num_loops: int = 3):
        super().__init__()
        self.num_loops = num_loops
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

        nn.init.eye_(self.recurrent_layer.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_loops):
            residual = hidden_states
            hidden_states = self.recurrent_layer(hidden_states)
            hidden_states = self.norm(hidden_states + residual)
        return hidden_states


# ---------------------------------------------------------------------------
# AnthosCore — full model
# ---------------------------------------------------------------------------

class AnthosCore(nn.Module):
    """
    Full Anthos model:
      base Qwen transformer → bifurcation layer → recurrent block → LM head

    Special token IDs are looked up at construction time from the tokenizer so the
    forward pass can detect <THT> and <IMG> positions without extra arguments.
    """

    def __init__(
        self,
        base_transformer: nn.Module,
        tokenizer,
        num_recurrent_loops: int = 3,
    ):
        super().__init__()
        self.transformer = base_transformer

        # Detect hidden size from base model config
        cfg = base_transformer.config
        hidden_size = cfg.hidden_size

        self.bifurcation = AnthosBifurcationLayer(hidden_size)
        self.recurrent = AnthosRecurrentBlock(hidden_size, num_loops=num_recurrent_loops)

        # Vision — wired in after construction via attach_vision()
        self.vision_encoder = None
        self.vision_projection = None

        # Special token IDs
        self.tht_id = tokenizer.convert_tokens_to_ids("<THT>")
        self.img_id = tokenizer.convert_tokens_to_ids("<IMG>")

        # LM head shared with transformer (tied embeddings)
        self.lm_head = base_transformer.lm_head

    def attach_vision(self, vision_encoder: nn.Module, vision_projection: nn.Module):
        """Wire in vision components transplanted from Qwen2.5-VL."""
        self.vision_encoder = vision_encoder
        self.vision_projection = vision_projection

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        # Build thought mask — True at positions where <THT> appeared in context
        thought_mask = (input_ids == self.tht_id)  # (batch, seq)

        # Vision path — interleave visual token embeddings
        inputs_embeds = None
        if images is not None and self.vision_encoder is not None:
            visual_tokens = self.vision_encoder(images)                    # (batch, vis_len, dim)
            visual_tokens = self.vision_projection(visual_tokens)          # project to hidden_size
            inputs_embeds = self._interleave_visual(input_ids, visual_tokens)

        # Main transformer pass
        transformer_out = self.transformer(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **{k: v for k, v in kwargs.items()
               if k not in ("output_hidden_states", "return_dict")},
            return_dict=True,
        )
        hidden = transformer_out.hidden_states[-1]  # (batch, seq, hidden)

        # Anthos layers
        hidden = self.bifurcation(hidden, thought_mask)
        hidden = self.recurrent(hidden)

        # LM head
        logits = self.lm_head(hidden)  # (batch, seq, vocab)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Return in HuggingFace CausalLMOutput style so Trainer works
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_out.past_key_values,
            hidden_states=transformer_out.hidden_states,
        )

    def _interleave_visual(self, input_ids, visual_tokens):
        """Replace <IMG> token embeddings with actual visual features."""
        base_embeds = self.transformer.get_input_embeddings()(input_ids)
        img_positions = (input_ids == self.img_id).nonzero(as_tuple=False)
        for pos in img_positions:
            b, s = pos[0].item(), pos[1].item()
            if b < visual_tokens.size(0):
                # Replace single IMG token with mean-pooled visual features
                base_embeds[b, s] = visual_tokens[b].mean(dim=0)
        return base_embeds

    # ------------------------------------------------------------------
    # Convenience wrappers so the Trainer can call generate / save / load
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self.transformer.config

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.transformer.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens):
        return self.transformer.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, path: str):
        """Save transformer weights + Anthos layers separately."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(str(path / "transformer"))
        torch.save({
            "bifurcation": self.bifurcation.state_dict(),
            "recurrent": self.recurrent.state_dict(),
        }, path / "anthos_layers.pt")
        print(f"Saved AnthosCore to {path}")

    @classmethod
    def from_pretrained(cls, path: str, tokenizer, **kwargs):
        """Load from a path written by save_pretrained."""
        path = Path(path)
        base = AutoModelForCausalLM.from_pretrained(
            str(path / "transformer"),
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        model = cls(base, tokenizer)
        layers_path = path / "anthos_layers.pt"
        if layers_path.exists():
            state = torch.load(layers_path, map_location="cpu")
            model.bifurcation.load_state_dict(state["bifurcation"])
            model.recurrent.load_state_dict(state["recurrent"])
        return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_anthos(
    tokenizer_path: str,
    base_model_id: str = "Qwen/Qwen2.5-9B-Instruct",
    checkpoint: str | None = None,
    device: str = "auto",
    num_recurrent_loops: int = 3,
) -> tuple:
    """
    Build or reload AnthosCore.

    Returns (model, tokenizer).

    If `checkpoint` is set, loads Anthos weights from that path.
    Otherwise loads the raw Qwen base and resizes embeddings for Anthos tokens.
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if checkpoint:
        print(f"Reloading AnthosCore from checkpoint {checkpoint}...")
        model = AnthosCore.from_pretrained(
            checkpoint, tokenizer,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    else:
        print(f"Loading base model {base_model_id}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        base.resize_token_embeddings(len(tokenizer))
        model = AnthosCore(base, tokenizer, num_recurrent_loops=num_recurrent_loops)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AnthosCore ready — {total/1e9:.2f}B params ({trainable/1e9:.2f}B trainable)")
    return model, tokenizer
