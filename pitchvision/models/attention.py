"""
Multi-head self-attention + transformer encoder.

Reference: "Attention Is All You Need" (Vaswani et al., 2017), Section 3.2.
Pre-norm variant follows Dosovitskiy et al. (2020).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Single QKV projection is a little faster than three Linears.
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (B, N, d_model)
            mask: (B, N) bool — True means "attend"; tokens where mask=False
                  are suppressed from being attended to.
        Returns:
            output:       (B, N, d_model)
            attn_weights: (B, num_heads, N, N)
        """
        B, N, _ = x.shape

        qkv = self.qkv(x)                                # (B, N, 3*d)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                 # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each (B, H, N, head_dim)

        # (B, H, N, N)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask over keys: shape (B, 1, 1, N) so it broadcasts across heads/queries.
            key_mask = mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                   # (B, H, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.proj_drop(self.proj(out))
        return out, attn


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)); then x + FFN(LN(x))."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class TacticalEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                 num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_per_layer = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attn_per_layer.append(attn)
        x = self.norm(x)
        return x, attn_per_layer
