"""
Loss functions for the tactical transformer.

- FocalLoss: class-imbalance-aware cross entropy (Lin et al., 2017).
- IntentTotalLoss: weighted sum over all prediction heads + rollout KL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.alpha if self.alpha.numel() > 0 else None
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class IntentTotalLoss(nn.Module):
    """
    Aggregates losses across all heads. Weights default to plan/finetune.yaml.
    """

    def __init__(self, intent_weights: torch.Tensor = None,
                 loss_weights: dict = None):
        super().__init__()
        self.focal = FocalLoss(gamma=2.0, alpha=intent_weights)
        w = loss_weights or {}
        self.w_dir = float(w.get("direction", 0.30))
        self.w_intent = float(w.get("intent", 0.30))
        self.w_urg = float(w.get("urgency", 0.10))
        self.w_pos = float(w.get("pos", 0.20))
        self.w_attn = float(w.get("attn", 0.10))

    def forward(self, pred: dict, target: dict) -> dict:
        B = target["intent"].shape[0]
        tp = target["target_idx"]
        if isinstance(tp, int):
            tp = torch.full((B,), tp, dtype=torch.long, device=target["intent"].device)
        idx = torch.arange(B, device=tp.device)

        l_dir = F.cross_entropy(pred["dir_logits"][idx, tp], target["direction"])
        l_intent = self.focal(pred["intent_logits"][idx, tp], target["intent"])
        l_urg = F.binary_cross_entropy_with_logits(
            torch.sigmoid(pred["urgency"][idx, tp]), target["urgency"])
        l_pos = F.smooth_l1_loss(pred["next_pos"][idx, tp], target["next_pos"])
        l_attn = F.kl_div(
            (pred["attn_rollout"] + 1e-8).log(),
            target["causal_dist"],
            reduction="batchmean",
        )

        total = (self.w_dir * l_dir + self.w_intent * l_intent
                 + self.w_urg * l_urg + self.w_pos * l_pos
                 + self.w_attn * l_attn)

        return {
            "total": total,
            "direction": l_dir.detach(),
            "intent": l_intent.detach(),
            "urgency": l_urg.detach(),
            "pos": l_pos.detach(),
            "attn": l_attn.detach(),
        }
