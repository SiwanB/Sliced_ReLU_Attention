import torch

def apply_rope(
            q,
            k,
            position_ids=None,       # None -> arange(T); or [T]; or [B, T]
            rope_base: float = 10000.0,
            rope_scale: float = 1.0, # >1 stretches to longer lengths 
            pos_offset: int = 0,     
        ):
        """
        q, k: [B, H, T, D] with even D
        returns: rotated (q, k) with the same shape/dtype/device
        """
        B, H, T, D = q.shape
        assert D % 2 == 0, f"RoPE requires even head_dim, got {D}"
        device, dtype = q.device, q.dtype

        # --- positions ---
        if position_ids is None:
            # [T]
            pos = torch.arange(T, device=device, dtype=torch.float32)
        else:
            # [T] or [B, T]
            pos = position_ids.to(device=device, dtype=torch.float32)

        if pos_offset:
            pos = pos + pos_offset
        if rope_scale != 1.0:
            pos = pos * rope_scale

        # --- frequencies (float32 for numerical stability) ---
        half = D // 2
        dim_ids = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = rope_base ** (-dim_ids / half)  # [half]

        # --- phase + sin/cos with correct broadcast shape ---
        if pos.dim() == 1:
            # pos: [T] -> phase: [T, half] -> sin/cos: [1,1,T,half]
            phase = pos[:, None] * inv_freq[None, :]
            sin = phase.sin().to(dtype)[None, None, :, :]
            cos = phase.cos().to(dtype)[None, None, :, :]
        elif pos.dim() == 2:
            # pos: [B, T] -> phase: [B, T, half] -> sin/cos: [B,1,T,half]
            phase = pos[..., None] * inv_freq[None, None, :]
            sin = phase.sin().to(dtype)[:, None, :, :]
            cos = phase.cos().to(dtype)[:, None, :, :]
        else:
            raise ValueError(f"position_ids must be [T] or [B, T], got shape {pos.shape}")

        # --- rotate even/odd channels ---
        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]      # [B,H,T,half] each
            return torch.cat([x1 * cos - x2 * sin,  # [B,H,T,half]
                            x1 * sin + x2 * cos], dim=-1)

        return rotate(q), rotate(k)