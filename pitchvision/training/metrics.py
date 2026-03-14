"""
Evaluation metrics for the tactical transformer.
"""

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_metrics(model, dataloader: DataLoader, device) -> dict:
    """
    Returns:
        intent_accuracy     — top-1 intent accuracy
        direction_acc_at_1  — correct within +/- 1 compass bin (tolerant)
        ade_m               — average displacement error on next_pos, meters
    """
    model.eval()

    n_correct_intent = 0
    n_correct_dir = 0
    n_total = 0
    ade_sum = 0.0

    for batch in dataloader:
        batch = _to_device(batch, device)
        pred = model(batch)

        B = batch["intent"].shape[0]
        tp = batch["target_idx"]
        if isinstance(tp, int):
            tp = torch.full((B,), tp, dtype=torch.long, device=device)
        idx = torch.arange(B, device=device)
        n_total += B

        intent_pred = pred["intent_logits"][idx, tp].argmax(dim=-1)
        n_correct_intent += (intent_pred == batch["intent"]).sum().item()

        dir_pred = pred["dir_logits"][idx, tp].argmax(dim=-1)
        diff = (dir_pred - batch["direction"]).abs()
        diff = torch.min(diff, 8 - diff)
        n_correct_dir += (diff <= 1).sum().item()

        pos_pred = pred["next_pos"][idx, tp]
        err = torch.norm(pos_pred - batch["next_pos"], dim=-1)
        ade_sum += err.sum().item()

    n = max(n_total, 1)
    return {
        "intent_accuracy": n_correct_intent / n,
        "direction_acc_at_1": n_correct_dir / n,
        "ade_m": ade_sum / n,
    }


def _to_device(batch, device):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
