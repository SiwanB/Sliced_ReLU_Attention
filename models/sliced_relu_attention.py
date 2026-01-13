# Sliced_ReLU_attention
import torch
import torch.nn as nn
from .rope import apply_rope

class SlicedReLUSelfAttention(nn.Module):
    def __init__(self, config, use_rope=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)


        self.proj_dropout = nn.Dropout(config.attention_probs_dropout_prob/2)

        self.use_rope = use_rope


        mid = self.all_head_size
        self.proj_layer = nn.Sequential(
            nn.Linear(self.all_head_size, mid),
            nn.GELU(),
            self.proj_dropout,
            nn.Linear(mid, self.num_attention_heads)
        )


    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)


    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        B, H, T, D = v.shape

        if self.use_rope:
            q, k = apply_rope(q, k)

        q_proj = self.proj_layer(q.reshape(B, T, -1)) #/ T
        k_proj = self.proj_layer(k.reshape(B, T, -1)) #/ T
        

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask[:, 0, :, 0]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, :, 0]
            
            # Expand for k_proj: [B, T, H]
            mask_exp = attention_mask.unsqueeze(-1).expand(-1, T, H)

            # Expand for v: [B, H, T, D]
            v = v * mask_exp.permute(0, 2, 1).unsqueeze(-1)
            
        else:
            attention_mask = torch.ones((B, T), device=q_proj.device, dtype=q_proj.dtype)
            mask_exp = torch.ones(B, T, H, device=v.device, dtype=v.dtype)
            
        # ---------------------------------------
        # 5. Zero q_proj / k_proj for padded tokens
        # (avoids padded projections polluting statistics)
        # ---------------------------------------
        proj_mask = attention_mask.unsqueeze(-1)  # [B,T,1]
        q_proj = q_proj * proj_mask
        k_proj = k_proj * proj_mask
    
        # ---------------------------------------
        # 6. Masked variance normalization
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

        Z = torch.cat((k_proj, q_proj), dim=1)
        Z = Z.view(B, 2 * T, H, 1).permute(0, 2, 1, 3)

        Z_sort, Z_sort_idx = torch.sort(Z, dim=2)
        Z_sort_idx_inv = torch.argsort(Z_sort_idx, dim=2)

        v_exp = torch.zeros((B, H, 2*T, D, 1), device=v.device, dtype=v.dtype)
        v_exp[:, :, :T] = (v - v.mean(dim=2, keepdim=True)).unsqueeze(-1)


        v_sorted = torch.gather(v_exp, dim=2, index=Z_sort_idx.unsqueeze(-2).expand(-1, -1, -1, D, 1))
        context = self.calc_eff_if_sorted(v_sorted, Z_sort)
        context = torch.gather(context, dim=2, index=Z_sort_idx_inv.unsqueeze(-2).expand(-1, -1, -1, D, 1))
        context = context[:, :, T:].squeeze(-1)


        # ---- Normalization term  Σ_j |q_i - k_j| ----
        # We use : |r| = 2 ReLU(r) - r
        # Since calc_eff_if_sorted already computes prefix sums of ReLU-like terms

        eps = 1e-4

        if attention_mask is not None:
            key_mask = mask_exp.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B,H,T,1,1)
        else:
            key_mask = torch.ones((B, H, T, 1, 1), device=v.device, dtype=v.dtype)
        v_norm = torch.zeros((B, H, 2*T, 1, 1), device=v.device, dtype=v.dtype)
        v_norm[:, :, :T] = key_mask  # 1 for keys, 0 for queries

        v_norm_sorted = torch.gather(v_norm, dim=2, index=Z_sort_idx.unsqueeze(-2).expand(-1, -1, -1, 1, 1))
        sum_relu = self.calc_eff_if_sorted(v_norm_sorted, Z_sort)  # shape (B,H,2T,1,1)
        
        
        total_v  = v_norm_sorted.sum(dim=2, keepdim=True)     # (B,H,1,1,1)
        total_vz = (v_norm_sorted * Z_sort.unsqueeze(-2)).sum(dim=2, keepdim=True)  # (B,H,1,1,1)
        norm_sorted = 2 * sum_relu - (Z_sort.unsqueeze(-2) * total_v - total_vz)    # (B,H,2T,1,1)
        norm_unsort = torch.gather(norm_sorted, dim=2, index=Z_sort_idx_inv.unsqueeze(-2).expand(-1, -1, -1, 1, 1))
        norm = norm_unsort[:, :, T:].squeeze(-1)  # shape (B,H,T,1)
        norm = torch.clamp(norm, min=eps)
        context = 2*context / norm
            
        norm_unsort = torch.gather(sum_relu, dim=2, index=Z_sort_idx_inv.unsqueeze(-2).expand(-1, -1, -1, 1, 1))
        norm = norm_unsort[:, :, T:].squeeze(-1)  # shape (B,H,T,1)
        norm = torch.clamp(norm, min=eps)
        context = context / (norm + eps)

        return context.transpose(1, 2).contiguous().view(B, T, self.all_head_size)

    
    def calc_eff_if_sorted(self, poids, Z):
        a = torch.cumsum(poids, dim=2)
        b = torch.cumsum(poids * Z.unsqueeze(-2), dim=2)
        return a * Z.unsqueeze(-2) - b