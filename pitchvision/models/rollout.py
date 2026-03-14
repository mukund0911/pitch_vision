"""
Attention rollout — composes attention matrices across layers so we can
measure end-to-end token-to-token influence.

Reference: Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers".

Algorithm per layer:
    A_hat = 0.5 * (average_over_heads(A) + I)      # account for residual
    A_hat = row_normalize(A_hat)
Composition: R = A_hat_L @ A_hat_{L-1} @ ... @ A_hat_1
R[i, j] = how much token i's final representation depends on token j.
"""

import torch


class AttentionRollout:
    def __init__(self, discard_residual_weight: float = 0.5):
        self.residual_weight = discard_residual_weight

    @torch.no_grad()
    def __call__(self, attn_weights_per_layer: list) -> torch.Tensor:
        """
        attn_weights_per_layer: list of (B, H, N, N) — one per layer.
        Returns: (B, N, N) rollout matrix.
        """
        B, H, N, _ = attn_weights_per_layer[0].shape
        device = attn_weights_per_layer[0].device
        eye = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)

        result = eye.expand(B, -1, -1).clone()
        for attn in attn_weights_per_layer:
            a = attn.mean(dim=1)  # (B, N, N)
            a = self.residual_weight * a + (1 - self.residual_weight) * eye
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            result = a @ result
        return result

    @staticmethod
    def ball_to_players(rollout: torch.Tensor, ball_idx: int,
                        player_idxs: list) -> torch.Tensor:
        """
        Pull out the slice R[:, ball_idx, player_idxs] and renormalize so it
        sums to 1 along the player axis.
        """
        influence = rollout[:, ball_idx, :][:, player_idxs]       # (B, N_players)
        influence = influence / influence.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return influence
