
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import rope


def repeat_kv(x, n_rep):
    """
    x: [B, n_kv_heads, T, head_dim]

    Returns:
        [B, n_heads, T, head_dim]
    """

    B, n_kv_heads, T, head_dim = x.shape

    if n_rep == 1:
        return x

    x = x[:, :, None, :, :]

    x = x.expand(
        B,
        n_kv_heads,
        n_rep,
        T,
        head_dim
    )

    return x.reshape(
        B,
        n_kv_heads * n_rep,
        T,
        head_dim
    )


def create_attention_mask(
    query_positions,
    key_positions,
    device,
    window_size: Optional[int] = None
):
    """
    query_positions: [T]
    key_positions:   [S]

    Returns:
        mask: [T, S]

    True  = attention is allowed
    False = attention is blocked
    """

    distance = (
        query_positions[:, None] -
        key_positions[None, :]
    )

    # Causal attention
    mask = distance >= 0

    # Optional sliding-window attention
    if window_size is not None:
        mask = mask & (distance < window_size)

    return mask.to(device=device)


class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.dropout = config.dropout
        self.bias = config.bias

        # Number of KV heads
        self.n_kv_heads = getattr(
            config,
            "n_kv_heads",
            None
        )

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # GQA repetition factor
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                "n_heads must be divisible by n_kv_heads."
            )

        self.n_rep = self.n_heads // self.n_kv_heads

        # RoPE configuration
        self.use_rope = (
            getattr(config, "rope_theta", None) is not None
        )

        # Sliding-window configuration
        self.window_size = getattr(
            config,
            "sliding_window",
            None
        )

        # Projections
        self.Wq = nn.Linear(
            self.d_model,
            self.n_heads * self.head_dim,
            bias=False
        )

        self.Wk = nn.Linear(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.Wv = nn.Linear(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.Wo = nn.Linear(
            self.n_heads * self.head_dim,
            self.d_model,
            bias=self.bias
        )

    def forward(
        self,
        x,
        cache=None,
        start_pos=None
    ):
        """
        x: [B, T, d_model]

        cache:
            None       -> no caching
            KVCache    -> standard growing cache
            RollingCache -> fixed-size rolling cache

        start_pos:
            Absolute position of the first input token.
        """

        B, T, C = x.shape

        if start_pos is None:
            start_pos = (
                cache.get_seq_length()
                if cache is not None
                else 0
            )

        # -----------------------------------------
        # 1. Q, K, V projections
        # -----------------------------------------

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # [B, T, heads, head_dim]
        Q = Q.view(
            B, T, self.n_heads, self.head_dim
        )

        K = K.view(
            B, T, self.n_kv_heads, self.head_dim
        )

        V = V.view(
            B, T, self.n_kv_heads, self.head_dim
        )

        # -----------------------------------------
        # 2. Apply RoPE
        # -----------------------------------------

        if self.use_rope:
            Q = rope(Q, start_pos=start_pos)
            K = rope(K, start_pos=start_pos)

        # -----------------------------------------
        # 3. Change layout
        # -----------------------------------------

        # [B, heads, T, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # -----------------------------------------
        # 4. Cache management
        # -----------------------------------------

        if cache is not None:

            # Cache stores the original KV heads.
            # It returns all relevant cached and new K/V.
            K, V, key_positions = cache.update(
                key=K,
                value=V,
                start_pos=start_pos
            )

        else:

            key_positions = torch.arange(
                start_pos,
                start_pos + T,
                device=x.device
            )

        # -----------------------------------------
        # 5. GQA: repeat K and V
        # -----------------------------------------

        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        # -----------------------------------------
        # 6. Attention scores
        # -----------------------------------------

        attn = torch.matmul(
            Q,
            K.transpose(-1, -2)
        )

        attn = attn / math.sqrt(self.head_dim)

        # -----------------------------------------
        # 7. Causal / sliding-window mask
        # -----------------------------------------

        query_positions = torch.arange(
            start_pos,
            start_pos + T,
            device=x.device
        )

        mask = create_attention_mask(
            query_positions=query_positions,
            key_positions=key_positions,
            device=x.device,
            window_size=self.window_size
        )

        # [1, 1, T, S] broadcasts across batch and heads
        attn = attn.masked_fill(
            ~mask[None, None, :, :],
            float("-inf")
        )

        # -----------------------------------------
        # 8. Softmax and dropout
        # -----------------------------------------

        attn = F.softmax(attn, dim=-1)

        attn = F.dropout(
            attn,
            p=self.dropout,
            training=self.training
        )

        # -----------------------------------------
        # 9. Weighted sum of values
        # -----------------------------------------

        out = torch.matmul(attn, V)

        # [B, heads, T, head_dim]
        # ->
        # [B, T, heads, head_dim]

        out = out.transpose(1, 2).contiguous()

        # Concatenate heads
        out = out.view(
            B,
            T,
            self.n_heads * self.head_dim
        )

        # Output projection
        out = self.Wo(out)

        return out
