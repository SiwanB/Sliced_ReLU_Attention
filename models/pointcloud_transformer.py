from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch.nn as nn

from .transformer import TransformerLayer
from .sliced_relu_attention import SlicedReLUSelfAttention
from .softmax_attention import SoftmaxSelfAttention
from .sliced_relu_bump_attention import SlicedReLUBumpSelfAttention


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