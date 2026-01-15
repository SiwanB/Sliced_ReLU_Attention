# Sliced_ReLU_attention
import torch
import torch.nn as nn
from .rope import apply_rope

class SlicedReLUSelfAttention(nn.Module):
    """
    Sliced ReLU attention (O(T log T) per head).

    - Projects Q/K to scalars z_q, z_k (one scalar per head and token).
    - Concatenates keys+queries, sorts along z, then uses windowed prefix sums to compute ReLU-attention efficiently.
    - Outputs standard multi-head context of shape (B, T, hidden_size).
    """

    def __init__(self, config, use_rope=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.proj_dropout = nn.Dropout(config.attention_probs_dropout_prob/2)

        self.use_rope = use_rope

        # Projection to one scalar per head (can be swapped for other designs).
        mid = self.all_head_size
        self.proj_layer = nn.Sequential(
            nn.Linear(self.all_head_size, mid),
            nn.GELU(),
            self.proj_dropout,
            nn.Linear(mid, self.num_heads)
        )


    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)


    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        B, H, T, D = v.shape

        # Positional encoding: RoPE
        if self.use_rope:
            q, k = apply_rope(q, k)

        # Project to scalars per head: q_proj, k_proj shape (B,T,H)
        q_proj = self.proj_layer(q.reshape(B, T, -1)) 
        k_proj = self.proj_layer(k.reshape(B, T, -1)) 
        
        # Attention mask handling: build mask_exp [B,T,H]
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask[:, 0, :, 0]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, :, 0]
            
            attention_mask = attention_mask.to(v.dtype)
            mask_exp = attention_mask.unsqueeze(-1).expand(-1, T, H)

            # Value masking
            v = v * mask_exp.permute(0, 2, 1).unsqueeze(-1)
            
        else:
            attention_mask = torch.ones((B, T), device=q_proj.device, dtype=q_proj.dtype)
            mask_exp = torch.ones(B, T, H, device=v.device, dtype=v.dtype)
            
        # Zero q_proj / k_proj for padded tokens
        proj_mask = attention_mask.unsqueeze(-1)  # [B,T,1]
        q_proj = q_proj * proj_mask
        k_proj = k_proj * proj_mask
    
        # ---------------------------------------
        # Masked centering + normalization for numerical stability.
        # ---------------------------------------
        eps = 1e-6

        # count real tokens per batch: [B,1,1]
        count = proj_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        # masked means
        mean_q = (q_proj.sum(dim=1, keepdim=True)) / count
        mean_k = (k_proj.sum(dim=1, keepdim=True)) / count

        # centered
        q_centered = q_proj - mean_q
        k_centered = k_proj - mean_k

        # masked variance
        var_q = ((q_centered ** 2).sum(dim=1, keepdim=True)) / count
        var_k = ((k_centered ** 2).sum(dim=1, keepdim=True)) / count

        # shared std
        std = torch.sqrt(0.5 * (var_q + var_k) + eps)

        # normalized
        q_proj = q_centered / std
        k_proj = k_centered / std


        # ------------------------
        # Sliced ReLU attention computations
        # ------------------------

        # Projected keys and queries are concatenated, to compute the attention output in one go
        Z = torch.cat((k_proj, q_proj), dim=1)
        Z = Z.view(B, 2 * T, H, 1).permute(0, 2, 1, 3)

        # Sorting of the concatened keys and queries, essential to the efficient sliced ReLU computations
        Z_sort, Z_sort_idx = torch.sort(Z, dim=2)
        Z_sort_idx_inv = torch.argsort(Z_sort_idx, dim=2)

        # Instantiate the shared value tensor. Values live on keys only and queries get zero values in the concatenation
        v_exp = torch.zeros((B, H, 2*T, D, 1), device=v.device, dtype=v.dtype)
        v_exp[:, :, :T] = (v - v.mean(dim=2, keepdim=True)).unsqueeze(-1)

        # Compute kernel on sorted inputs, then unsort and keep the query part
        v_sorted = torch.gather(v_exp, dim=2, index=Z_sort_idx.unsqueeze(-2).expand(-1, -1, -1, D, 1))
        context = self.calc_eff_if_sorted(v_sorted, Z_sort)
        context = torch.gather(context, dim=2, index=Z_sort_idx_inv.unsqueeze(-2).expand(-1, -1, -1, D, 1))
        context = context[:, :, T:].squeeze(-1)


        # ------------------
        # Normalization term  Σ_j |q_i - k_j|. We use : |r| = 2 ReLU(r) - r and calc_eff_if_sorted. No need to re-sort Z.
        # ------------------

        eps = 1e-4

        if attention_mask is not None:
            key_mask = mask_exp.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B,H,T,1,1)
        else:
            key_mask = torch.ones((B, H, T, 1, 1), device=v.device, dtype=v.dtype)
        v_norm = torch.zeros((B, H, 2*T, 1, 1), device=v.device, dtype=v.dtype)
        v_norm[:, :, :T] = key_mask  # 1 for keys, 0 for queries

        # Compute the normalization term, using the same logic than previously.
        v_norm_sorted = torch.gather(v_norm, dim=2, index=Z_sort_idx.unsqueeze(-2).expand(-1, -1, -1, 1, 1))
        sum_relu = self.calc_eff_if_sorted(v_norm_sorted, Z_sort)  # shape (B,H,2T,1,1)
        
        # Correcting terms to get |.| instead of ReLU(.)
        total_v  = v_norm_sorted.sum(dim=2, keepdim=True)     # (B,H,1,1,1)
        total_vz = (v_norm_sorted * Z_sort.unsqueeze(-2)).sum(dim=2, keepdim=True)  # (B,H,1,1,1)
        norm_sorted = 2 * sum_relu - (Z_sort.unsqueeze(-2) * total_v - total_vz)    # (B,H,2T,1,1)

        # Same unsorting logic than previously.
        norm_unsort = torch.gather(norm_sorted, dim=2, index=Z_sort_idx_inv.unsqueeze(-2).expand(-1, -1, -1, 1, 1))
        norm = norm_unsort[:, :, T:].squeeze(-1)  # shape (B,H,T,1)
        norm = torch.clamp(norm, min=eps)
        context = 2*context / norm


        return context.transpose(1, 2).contiguous().view(B, T, self.all_head_size)

    
    def calc_eff_if_sorted(self, poids, Z):
        """
        Efficient prefix-sum primitive for sorted scalars Z.
        For each position i: computes sum_{j<=i} (Z_i - Z_j) * weights_j
        (broadcasted over value dimensions).
        """
        a = torch.cumsum(poids, dim=2)
        b = torch.cumsum(poids * Z.unsqueeze(-2), dim=2)
        return a * Z.unsqueeze(-2) - b