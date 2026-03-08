"""
PlayerToken — fuse position + kinematics + role into a single 128-d embedding.

Modality widths are chosen so they sum to d_model=128:
    position (43) + kinematics (43) + role (42) = 128
"""

import torch
import torch.nn as nn

from .encodings import FourierPositionalEncoding


class PlayerToken(nn.Module):
    POS_DIM = 43
    KIN_DIM = 43
    ROLE_DIM = 42

    def __init__(self, d_model: int = 128, num_roles: int = 7,
                 kinematic_dim: int = 4, fourier_bands: int = 8):
        super().__init__()
        assert self.POS_DIM + self.KIN_DIM + self.ROLE_DIM == d_model, \
            "modality dims must sum to d_model"

        self.pos_enc = FourierPositionalEncoding(
            in_dim=2, out_dim=self.POS_DIM, num_bands=fourier_bands)
        self.kin_proj = nn.Linear(kinematic_dim, self.KIN_DIM)
        self.role_emb = nn.Embedding(num_roles, self.ROLE_DIM)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pos: torch.Tensor, kinematics: torch.Tensor,
                role: torch.Tensor) -> torch.Tensor:
        """
        pos:        (..., 2)
        kinematics: (..., 4)
        role:       (...) long
        returns:    (..., d_model)
        """
        p = self.pos_enc(pos)
        k = self.kin_proj(kinematics)
        r = self.role_emb(role)
        return self.norm(torch.cat([p, k, r], dim=-1))
