import torch
import torch.nn as nn

from .rope import apply_rope

class SoftmaxSelfAttention(nn.Module):
    def __init__(self, config, use_rope=False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.use_rope = use_rope


    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        if self.use_rope:
            q, k = apply_rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            scores += attention_mask.to(dtype=scores.dtype)
        attention_probs = torch.nn.functional.softmax(scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, v)
        return context.transpose(1, 2).contiguous().view(hidden_states.size(0), -1, self.all_head_size)