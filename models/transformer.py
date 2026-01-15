"""
Transformer encoder compatible with Hugging Face, with pluggable attention kernels.

- Uses RoBERTa embeddings / config for easy HF integration.
- Replaces the standard RobertaEncoder stack with custom layers.
- Returns BaseModelOutput(last_hidden_state=...).
"""


from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn

from .sliced_relu_attention import SlicedReLUSelfAttention
from .softmax_attention import SoftmaxSelfAttention
from .sliced_relu_bump_attention import SlicedReLUBumpSelfAttention


class TransformerLayer(nn.Module):
    """Pre-LN Transformer block: LN -> Attention -> residual, then LN -> FFN -> residual."""

    def __init__(self, config, attention_module):
        super().__init__()
        self.attn = attention_module
        self.attn_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = self.attn_dropout(self.attn_dense(x)) + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return self.ffn_dropout(x) + residual
    

class TransformerEncoder(RobertaPreTrainedModel):
    """
    HF-compatible encoder that reuses RoBERTa embeddings but swaps the attention mechanism.

    Config fields used:
      - attention_type: {"softmax", "sliced_relu", "sliced_relu_bump"}
      - use_rope: bool (if True, disable absolute position embeddings)
    """

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False) 

        self.attention_type = getattr(config, "attention_type", "softmax")

        self.use_rope = bool(getattr(config, "use_rope", False))

        # --- Neutralize absolute positional embeddings when using RoPE ---
        if self.use_rope:
            pos_emb = self.roberta.embeddings.position_embeddings
            # Freeze and zero: embeddings still get added, but they add 0 → no absolute PE
            pos_emb.weight.requires_grad = False
            with torch.no_grad():
                pos_emb.weight.zero_()
        # ------

        # Remove unused Roberta encoder, we provide our own encoder stack below
        del self.roberta.encoder

        self.encoder = nn.ModuleList()

        for _ in range(config.num_hidden_layers):
            if self.attention_type == "softmax":
                attention = SoftmaxSelfAttention(config, use_rope=self.use_rope)
            elif self.attention_type == "sliced_relu":
                attention = SlicedReLUSelfAttention(config, use_rope=self.use_rope)
            elif self.attention_type == "sliced_relu_bump":
                attention = SlicedReLUBumpSelfAttention(config, use_rope=self.use_rope)
            else:
                raise ValueError(f"Unknown attention_type={self.attention_type!r}! Use softmax, sliced_relu, or sliced_relu_bump.")

            self.encoder.append(TransformerLayer(config, attention))

        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def _embed_no_abs_pos(self, input_ids):
        # Reproduce RobertaEmbeddings but without adding position_embeddings
        # Needed when use_rope=True
        # Manual path: word → (token_type if present) → LN → dropout

        emb = self.roberta.embeddings
        x = emb.word_embeddings(input_ids)                     # [B, T, H]

        # RoBERTa normally has no token_type; keep this guard for safety.
        if hasattr(emb, "token_type_embeddings") and emb.token_type_embeddings is not None:
            token_type_ids = torch.zeros_like(input_ids)
            x = x + emb.token_type_embeddings(token_type_ids)

        x = emb.LayerNorm(x)
        return emb.dropout(x)

    def forward(self, input_ids, attention_mask=None):
        
        if self.use_rope:
            # Bypass RobertaEmbeddings.forward to avoid absolute PEs entirely
            hidden_states = self._embed_no_abs_pos(input_ids)
        else:
            # Standard path (includes absolute positional embeddings)
            hidden_states = self.roberta.embeddings(input_ids=input_ids)

        if attention_mask is not None:
            if self.attention_type == "softmax":
                # Softmax-style attention: transform to large negative mask
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_layernorm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)