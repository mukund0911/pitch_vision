"""
Positional encodings.

FourierPositionalEncoding — random Fourier features (Tancik et al., 2020).
FrameEmbedding            — learnable temporal embedding over frame index.
"""

import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """
    Maps continuous coordinates to a high-dimensional embedding using random
    Fourier features, then projects to the desired output size.

        features = [sin(2π · x · B), cos(2π · x · B)]          -> 2*num_bands dims
        output   = Linear(features)                            -> out_dim
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 32, num_bands: int = 8,
                 scale: float = 2.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        # Learnable frequency matrix — let the model learn its own bandwidth.
        self.B = nn.Parameter(torch.randn(in_dim, num_bands) * scale)
        self.proj = nn.Linear(num_bands * 2, out_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (..., in_dim)
        returns: (..., out_dim)
        """
        projected = 2.0 * math.pi * (coords @ self.B)   # (..., num_bands)
        features = torch.cat([projected.sin(), projected.cos()], dim=-1)
        return self.proj(features)


class FrameEmbedding(nn.Module):
    def __init__(self, num_frames: int = 5, d_model: int = 128):
        super().__init__()
        self.table = nn.Embedding(num_frames, d_model)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        return self.table(frame_indices)
