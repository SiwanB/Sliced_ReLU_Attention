from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, RobertaPreTrainedModel#, RobertaEmbeddings
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn

from .sliced_relu_attention import *
from .softmax_attention import *
from .sliced_relu_bump_attention import *


class TransformerLayer(nn.Module):
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

        # Remove unused Roberta encoder
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
        
        
        
class PointCloudTransformerEncoder(RobertaPreTrainedModel):
    """
    Point-wise transformer backbone for point clouds, replacing token embeddings with a point MLP: (x, y, z) -> hidden_size.

    Inputs:
      points: (B, N, 3)
      attention_mask: optional (B, N)
    """

    def __init__(self, config):
        super().__init__(config)

        self.use_rope = bool(getattr(config, "use_rope", False))

        self.attention_type = getattr(config, "attention_type", "softmax")

        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps
        dropout_p = config.hidden_dropout_prob

        # ----- Point-wise embedding: (x,y,z) → hidden_size -----
        self.point_embedding = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.input_dropout = nn.Dropout(dropout_p)

        # ----- Encoder stack (reuses your CustomRobertaLayer + attention modules) -----
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

        self.final_layernorm = nn.LayerNorm(hidden_size, eps=ln_eps)

        # Hugging Face weight init
        self.post_init()

    def forward(self, points, attention_mask=None):
        """
        points: (B, N, 3)
        attention_mask: optional (B, N) with 1=keep, 0=mask (like HF)
        returns: (B, N, hidden_size)
        """

        # Point-wise embedding
        # (B, N, 3) -> (B, N, H)
        hidden_states = self.point_embedding(points)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.input_dropout(hidden_states)

        # Attention mask formatting
        if attention_mask is not None:
            if self.attention_type == "softmax":
                # Softmax-style attention: transform to large negative mask
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        # Encoder stack 
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layernorm 
        hidden_states = self.final_layernorm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)