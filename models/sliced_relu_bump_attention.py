#Sliced ReLU-bump attention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope


class SlicedReLUBumpSelfAttention(nn.Module):
    """
    Sliced ReLU-bump attention (O(T log T) per head).

    - Projects Q/K to scalars z_q, z_k (one scalar per head and token).
    - Concatenates keys+queries, sorts along z, then uses prefix sums to compute ReLU-attention efficiently.
    - Outputs standard multi-head context of shape (B, T, hidden_size).
    """
    def __init__(self, config, use_rope: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.use_rope = use_rope

        # Projection to one scalar per head (can be swapped for other designs).
        self.proj_layer = nn.Linear(self.all_head_size, self.num_heads, bias=False)

        # Per-head bandwidth parameter: b_h = softplus(log_bw_h) + eps > 0. Init around ~0.7 (softplus(0))
        self.log_bandwidth = nn.Parameter(torch.zeros(self.num_heads))
        self.bandwidth_eps = 1e-4


    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)
   
   
    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))  # (B,H,T,D)
        k = self.transpose_for_scores(self.key(hidden_states))    # (B,H,T,D)
        v = self.transpose_for_scores(self.value(hidden_states))  # (B,H,T,D)

        B, H, T, D = v.shape

        # Positional encoding: RoPE
        if self.use_rope:
            q, k = apply_rope(q, k)

        # Project to scalars per head: q_proj, k_proj shape (B,T,H)
        q_proj = self.proj_layer(q.reshape(B, T, -1))  # (B, T, H)
        k_proj = self.proj_layer(k.reshape(B, T, -1))  # (B, T, H)


        # Attention mask handling: build mask_exp [B,T,H]
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask[:, 0, :, 0]   # [B,T]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, 0, :]
                
            attention_mask = attention_mask.to(v.dtype)
            mask_exp = attention_mask.unsqueeze(-1).expand(-1, T, H)  # [B,T,H]

            # Value masking
            v = v * mask_exp.permute(0, 2, 1).unsqueeze(-1)
        else:
            mask_exp = torch.ones(B, T, H, device=v.device, dtype=v.dtype)
            attention_mask = torch.ones((B, T), device=q_proj.device, dtype=q_proj.dtype)


        # ------------------------
        # Sliced ReLU-bump attention computations
        # ------------------------

        # Projected keys and queries are concatenated, to compute the attention output in one go
        Z = torch.cat((k_proj, q_proj), dim=1)          # (B, 2T, H)
        Z = Z.view(B, 2 * T, H, 1).permute(0, 2, 1, 3)  # (B,H,2T,1)

        Z_sort, Z_sort_idx = torch.sort(Z.to(torch.float32), dim=2)       # (B,H,2T,1)
        Z_sort_idx = Z_sort_idx.squeeze(-1)             # (B,H,2T)
        Z_sort_idx_inv = torch.argsort(Z_sort_idx, dim=2)  # (B,H,2T)

        # Instantiate the shared value tensor. Values live on keys only and queries get zero values in the concatenation
        zero_keys = torch.zeros_like(v)                         # (B,H,T,D)
        v_exp = torch.cat([v, zero_keys], dim=2)[..., None]     # (B,H,2T,D,1)

        # Compute kernel on sorted inputs, then unsort and keep the query part
        idx_v = Z_sort_idx[..., None, None].expand(-1, -1, -1, D, 1)  # (B,H,2T,D,1)
        v_sorted = v_exp.gather(2, idx_v)                             # (B,H,2T,D,1)
        bw_param = self.log_bandwidth  # (H,)
        context_sorted = self.calc_bump_if_sorted(v_sorted, Z_sort, bw_param)  # (B,H,2T,D,1)
        idx_ctx = Z_sort_idx_inv[..., None, None].expand(-1, -1, -1, D, 1)     # (B,H,2T,D,1)
        context_unsort = context_sorted.gather(2, idx_ctx)                     # (B,H,2T,D,1)

        context = context_unsort[:, :, T:].squeeze(-1)  # (B,H,T,D)

        # ------------------
        # Simple normalization term equal to the number of valid tokens (keeps scale roughly length-invariant)
        # ------------------
        real_lens = attention_mask.sum(dim=1, dtype=torch.float32)            # (B,)
        real_lens = torch.clamp(real_lens, min=1).view(B, 1, 1, 1)
        context = context / real_lens.to(context.dtype)


        return context.transpose(1, 2).contiguous().view(B, T, self.all_head_size)
   
    def calc_bump_if_sorted(self, v_sorted, Z_sort, bandwidth_per_head):
        """
        v_sorted:  (B, H, L, D, 1)
        Z_sort:    (B, H, L, 1)   sorted scalar projections
        bandwidth_per_head: (H,) or (B,H)

        Returns:
            out: (B, H, L, D, 1)

        Implements, for each head h and position i:

            out_i = sum_j f_h(z_i - z_j) * v_j,

        where
            f_h(r) = max(0, 1 - |r| / b_h).

        Uses searchsorted to get the window [z_i - b, z_i + b] and prefix sums of v and z*v
        to evaluate the piecewise linear kernel in linear time once z is sorted. 
        """
        
        B, H, L, D, _ = v_sorted.shape
        BH = B * H

        orig_dtype = v_sorted.dtype

        # Flatten (B,H) -> (BH) and recast to fp32 for numerical stability
        v = v_sorted.squeeze(-1).reshape(BH, L, D).to(torch.float32)
        z = Z_sort.reshape(BH, L).to(torch.float32)

        # Per-head bandwidth b_h > 0
        if bandwidth_per_head.dim() == 1:  # (H,)
            bw = bandwidth_per_head.unsqueeze(0).expand(B, H).reshape(BH)  # (BH,)
        else:  # (B,H)
            bw = bandwidth_per_head.reshape(BH)

        bw = bw.to(torch.float32)
        bw = F.softplus(bw) + float(self.bandwidth_eps)       # ensure > 0
        bw_col = bw.unsqueeze(-1)                      # (BH, 1)

        # Prefix sums over sorted z: P_v[k] = sum_{j <= k} v_j, P_zv[k] = sum_{j <= k} z_j v_j
        P_v  = v.cumsum(dim=1)                             # (BH, L, D)
        P_zv = (v * z.unsqueeze(-1)).cumsum(dim=1)         # (BH, L, D)

        # Pad with zero at start so that: sum_{j=a..b} = P[b+1]-P[a]
        zero_v   = torch.zeros(BH, 1, D, dtype=v.dtype, device=v.device)
        P_v_pad  = torch.cat([zero_v,  P_v],  dim=1)       # (BH, L+1, D)
        P_zv_pad = torch.cat([zero_v, P_zv], dim=1)        # (BH, L+1, D)

        # Window bounds: z_j in [z_i - b, z_i + b]
        z_minus = z - bw_col           # (BH, L)
        z_plus  = z + bw_col           # (BH, L)

        # Window indices per i: left = first z>=z_i-b, rightp1 = first z>z_i+b
        left_idx   = torch.searchsorted(z, z_minus, right=False)  # (BH, L) first j with z_j >= z_i - b
        rightp1_idx = torch.searchsorted(z, z_plus,  right=True)   # (BH, L) first j with z_j >  z_i + b (=> R+1)

        # For convenience: P[:, i] corresponds to sum up to index i (inclusive) in the unpadded array
        #   Pv_i  = P_v_pad[:, 1:]   (sum_{j <= i} v_j)
        #   Pzv_i = P_zv_pad[:, 1:]
        Pv_i  = P_v_pad[:, 1:]      # (BH, L, D)
        Pzv_i = P_zv_pad[:, 1:]     # (BH, L, D)

        # Expand indices to gather D-dimensional prefix sums
        left_idx_exp   = left_idx.unsqueeze(-1).expand(-1, -1, D)     # (BH, L, D)
        rightp1_idx_exp = rightp1_idx.unsqueeze(-1).expand(-1, -1, D)   # (BH, L, D)

        # Prefix sums at window boundaries: P[L], P[R+1]
        Pv_left    = P_v_pad.gather(1, left_idx_exp)      # sum_{j < L(i)} v_j
        Pv_rightp1  = P_v_pad.gather(1, rightp1_idx_exp)    # sum_{j <= R(i)} v_j

        Pzv_left   = P_zv_pad.gather(1, left_idx_exp)
        Pzv_rightp1 = P_zv_pad.gather(1, rightp1_idx_exp)

        # Segment sums:
        # left  = Σ_{j in [L..i]} (·),   right = Σ_{j in [i+1..R]} (·)
        sum_v_left  = Pv_i  - Pv_left        # (BH, L, D)
        sum_zv_left = Pzv_i - Pzv_left       # (BH, L, D)

        sum_v_right  = Pv_rightp1 - Pv_i      # (BH, L, D)
        sum_zv_right = Pzv_rightp1 - Pzv_i    # (BH, L, D)

        # Compute z_i/b and 1/b
        inv_bw    = (1.0 / bw_col).unsqueeze(-1)   # (BH, 1, 1)
        z_over_bw = (z / bw_col).unsqueeze(-1)     # (BH, L, 1), this is z_i / b_h

        # Closed-form on each side using sums of v_j and z_j v_j:
        # left  (j <= i): (1 - z_i/b) * Σv + (1/b) * Σ(zv)
        # right (j >  i): (1 + z_i/b) * Σv - (1/b) * Σ(zv)
        left  = sum_v_left  * (1.0 - z_over_bw) + sum_zv_left  * inv_bw
        right = sum_v_right * (1.0 + z_over_bw) - sum_zv_right * inv_bw

        # Combine both sides
        out = left + right   # (BH, L, D)

        # Reshape
        out = out.view(B, H, L, D).unsqueeze(-1)
        return out.to(orig_dtype)