"""
IntentDecoder — 4 prediction heads sharing the player token as input.
"""

import torch
import torch.nn as nn


def _mlp(d_model: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class IntentDecoder(nn.Module):
    def __init__(self, d_model: int = 128, num_directions: int = 8,
                 num_intents: int = 5):
        super().__init__()
        self.direction_head = _mlp(d_model, num_directions)
        self.intent_head = _mlp(d_model, num_intents)
        self.urgency_head = _mlp(d_model, 1)
        self.pos_head = _mlp(d_model, 2)

    def forward(self, player_tokens: torch.Tensor) -> dict:
        """
        player_tokens: (B, N, d_model)
        Returns dict with:
            dir_logits:    (B, N, num_directions)
            intent_logits: (B, N, num_intents)
            urgency:       (B, N)
            next_pos:      (B, N, 2)
        """
        dir_logits = self.direction_head(player_tokens)
        intent_logits = self.intent_head(player_tokens)
        urgency = torch.sigmoid(self.urgency_head(player_tokens).squeeze(-1))
        next_pos = self.pos_head(player_tokens)
        return {
            "dir_logits": dir_logits,
            "intent_logits": intent_logits,
            "urgency": urgency,
            "next_pos": next_pos,
        }
