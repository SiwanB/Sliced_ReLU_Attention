import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope


class SlicedReLUBumpSelfAttention(nn.Module):
    def __init__(self, config, use_rope: bool = False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.use_rope = use_rope

        # Projection from all heads to one scalar per head
        self.proj_layer = nn.Linear(self.all_head_size, self.num_attention_heads, bias=False)

        # Per-head bandwidth parameter: b_h = softplus(log_bw_h) + eps > 0
        # Init around ~0.7 (softplus(0))
        self.log_bandwidth = nn.Parameter(torch.zeros(self.num_attention_heads))
        self.bandwidth_eps = 1e-4

   # -------------------- helpers --------------------

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)
   

   # -------------------- forward --------------------
   
    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))  # (B,H,T,D)
        k = self.transpose_for_scores(self.key(hidden_states))    # (B,H,T,D)
        v = self.transpose_for_scores(self.value(hidden_states))  # (B,H,T,D)
        B, H, T, D = v.shape

        if self.use_rope:
            q, k = apply_rope(q, k)

        # Project to scalars per head: q_proj, k_proj shape (B,T,H)
        q_proj = self.proj_layer(q.reshape(B, T, -1))  # (B, T, H)
        k_proj = self.proj_layer(k.reshape(B, T, -1))  # (B, T, H)


        # --- attention mask handling: build mask_exp [B,T,H] ---
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask[:, 0, :, 0]   # [B,T]
                #attention_mask = attention_mask[:, 0, 0, :]
            elif attention_mask.dim() == 3:
                #attention_mask = attention_mask[:, :, 0]      # [B,T]
                attention_mask = attention_mask[:, 0, :]
                
            attention_mask = attention_mask.to(v.dtype)
            mask_exp = attention_mask.unsqueeze(-1).expand(-1, T, H)  # [B,T,H]
            # Zero out values for masked tokens
            v = v * mask_exp.permute(0, 2, 1).unsqueeze(-1)
        else:
            mask_exp = torch.ones(B, T, H, device=v.device, dtype=v.dtype)
            attention_mask = torch.ones((B, T), device=q_proj.device, dtype=q_proj.dtype)


        # --- build concatenated Z for keys+queries ---
        # Z: (B, 2T, H) -> (B,H,2T,1)
        Z = torch.cat((k_proj, q_proj), dim=1)          # (B, 2T, H)
        Z = Z.view(B, 2 * T, H, 1).permute(0, 2, 1, 3)  # (B,H,2T,1)

        Z_sort, Z_sort_idx = torch.sort(Z, dim=2)       # (B,H,2T,1)
        Z_sort_idx = Z_sort_idx.squeeze(-1)             # (B,H,2T)
        Z_sort_idx_inv = torch.argsort(Z_sort_idx, dim=2)  # (B,H,2T)

        # --- values: keys have v, queries have value 0 ---
        zero_keys = torch.zeros_like(v)                         # (B,H,T,D)
        v_exp = torch.cat([v, zero_keys], dim=2)[..., None]     # (B,H,2T,D,1)

        # sort values by the same permutation
        idx_v = Z_sort_idx[..., None, None].expand(-1, -1, -1, D, 1)  # (B,H,2T,D,1)
        v_sorted = v_exp.gather(2, idx_v)                             # (B,H,2T,D,1)

        # --- bump attention for the context ---
        bw_param = self.log_bandwidth  # (H,)
        context_sorted = self.calc_bump_if_sorted(v_sorted, Z_sort, bw_param)  # (B,H,2T,D,1)

        # unsort and keep only the query part (last T positions in original concatenation)
        idx_ctx = Z_sort_idx_inv[..., None, None].expand(-1, -1, -1, D, 1)     # (B,H,2T,D,1)
        context_unsort = context_sorted.gather(2, idx_ctx)                     # (B,H,2T,D,1)

        context = context_unsort[:, :, T:].squeeze(-1)  # (B,H,T,D)


        real_lens = attention_mask.sum(dim=1)            # (B,)
        real_lens = real_lens.view(B, 1, 1, 1)           # (B,1,1,1)
        real_lens = torch.clamp(real_lens, min=1)
        context = context / real_lens

        # back to [B,T,hidden]
        return context.transpose(1, 2).contiguous().view(B, T, self.all_head_size)
   
    def calc_bump_if_sorted(self, v_sorted, Z_sort, bandwidth_per_head):
        """
        v_sorted:  (B, H, L, D, 1)
        Z_sort:    (B, H, L, 1)   sorted scalars z_j
        bandwidth_per_head: (H,) or (B,H)

        Returns:
            out: (B, H, L, D, 1)

        Implements, for each head h and position i:

            out_i = sum_j f_h(z_i - z_j) * v_j,

        where
            f_h(r) = max(0, 1 - |r| / b_h).

        This is done via:
            - prefix sums over v_j and z_j v_j
            - searchsorted to get the local window [z_i - b_h, z_i + b_h]
            - closed-form formulas for the linear pieces on left/right.
        """
        B, H, L, D, _ = v_sorted.shape
        BH = B * H

        # Flatten (B,H) -> (BH)
        # v: (BH, L, D), z: (BH, L)
        v = v_sorted.squeeze(-1).reshape(BH, L, D)
        z = Z_sort.reshape(BH, L).to(torch.float32)

        # Per-head bandwidth b_h > 0
        if bandwidth_per_head.dim() == 1:  # (H,)
            bw = bandwidth_per_head.unsqueeze(0).expand(B, H).reshape(BH)  # (BH,)
        else:  # (B,H)
            bw = bandwidth_per_head.reshape(BH)

        bw = F.softplus(bw) + self.bandwidth_eps       # ensure > 0
        bw = torch.clamp(bw, min=self.bandwidth_eps)
        bw_col = bw.unsqueeze(-1)                      # (BH, 1)

        # Prefix sums: P_v[k] = sum_{j <= k} v_j
        #              P_zv[k] = sum_{j <= k} z_j v_j
        P_v  = v.cumsum(dim=1)                             # (BH, L, D)
        P_zv = (v * z.unsqueeze(-1)).cumsum(dim=1)         # (BH, L, D)

        # Pad with zero at index 0 for convenient [a,b] segments: sum_{j=a..b} v_j = P[b+1]-P[a]
        zero_v   = torch.zeros(BH, 1, D, dtype=v.dtype, device=v.device)
        P_v_pad  = torch.cat([zero_v,  P_v],  dim=1)       # (BH, L+1, D)
        P_zv_pad = torch.cat([zero_v, P_zv], dim=1)        # (BH, L+1, D)

        # Window bounds: z_j in [z_i - b, z_i + b]
        z_minus = z - bw_col           # (BH, L)
        z_plus  = z + bw_col           # (BH, L)

        # searchsorted row-wise; returns indices in [0, L]
        L_idx   = torch.searchsorted(z, z_minus, right=False)  # (BH, L) first j with z_j >= z_i - b
        Rp1_idx = torch.searchsorted(z, z_plus,  right=True)   # (BH, L) first j with z_j >  z_i + b (=> R+1)

        # We avoid building an explicit idx/ip1 tensor:
        # i indices correspond to P_v_pad[:, 1:] for sums up to i.
        # So:
        #   Pv_i  = P_v_pad[:, 1:]   (sum_{j <= i} v_j)
        #   Pzv_i = P_zv_pad[:, 1:]
        Pv_i  = P_v_pad[:, 1:]      # (BH, L, D)
        Pzv_i = P_zv_pad[:, 1:]     # (BH, L, D)

        # Expand L_idx, Rp1_idx to (BH, L, D) for gather
        L_idx_exp   = L_idx.unsqueeze(-1).expand(-1, -1, D)     # (BH, L, D)
        Rp1_idx_exp = Rp1_idx.unsqueeze(-1).expand(-1, -1, D)   # (BH, L, D)

        # Segment sums via prefix sums
        Pv_L    = P_v_pad.gather(1, L_idx_exp)      # sum_{j < L(i)} v_j
        Pv_Rp1  = P_v_pad.gather(1, Rp1_idx_exp)    # sum_{j <= R(i)} v_j

        Pzv_L   = P_zv_pad.gather(1, L_idx_exp)
        Pzv_Rp1 = P_zv_pad.gather(1, Rp1_idx_exp)

        # left window: j ? [L(i), i]
        sum_v_left  = Pv_i  - Pv_L        # (BH, L, D)
        sum_zv_left = Pzv_i - Pzv_L       # (BH, L, D)

        # right window: j ? [i+1, R(i)]  => P(R+1) - P(i+1)
        sum_v_right  = Pv_Rp1 - Pv_i      # (BH, L, D)
        sum_zv_right = Pzv_Rp1 - Pzv_i    # (BH, L, D)

        # Coefficients depending on z_i / b_h
        inv_bw    = (1.0 / bw_col).unsqueeze(-1)   # (BH, 1, 1)
        z_over_bw = (z / bw_col).unsqueeze(-1)     # (BH, L, 1), this is z_i / b_h

        # Correct piecewise formulas:
        #  For j <= i: f = 1 - (z_i - z_j)/b = (1 - z_i/b) + (z_j/b)
        left  = sum_v_left  * (1.0 - z_over_bw) + sum_zv_left  * inv_bw

        #  For j > i:  f = 1 - (z_j - z_i)/b = (1 + z_i/b) - (z_j/b)
        right = sum_v_right * (1.0 + z_over_bw) - sum_zv_right * inv_bw

        out = left + right   # (BH, L, D)

        # Reshape back to (B,H,L,D,1)
        out = out.view(B, H, L, D).unsqueeze(-1)
        return out