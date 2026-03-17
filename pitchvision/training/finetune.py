"""
Supervised fine-tuning loop for PitchVisionModel.

Minimal trainer — no distributed training, no mixed precision (can be added
later). Reads configs/finetune.yaml, trains for N epochs, saves the best
checkpoint by intent accuracy on the validation split.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pitchvision.data.augmentation import TacticalAugmentation
from pitchvision.data.dataset import SoccerIntentDataset
from pitchvision.models.pitchvision import PitchVisionModel
from pitchvision.training.losses import IntentTotalLoss
from pitchvision.training.metrics import compute_metrics


def train(cfg, dry_run: bool = False):
    torch.manual_seed(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PitchVisionModel(cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    pretrain_ckpt = cfg.training.get("pretrain_ckpt")
    if pretrain_ckpt:
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded pretrain: {pretrain_ckpt}")

    augment = TacticalAugmentation()
    train_ds = SoccerIntentDataset(
        cfg.data.episodes_dir, split="train", augment=augment,
        num_frames=int(cfg.model.num_frames),
        num_players=int(cfg.model.num_players),
        kinematic_dim=int(cfg.token.kinematic_dim),
        field_length=float(cfg.data.field_length),
        field_width=float(cfg.data.field_width),
    )
    val_ds = SoccerIntentDataset(
        cfg.data.episodes_dir, split="val", augment=None,
        num_frames=int(cfg.model.num_frames),
        num_players=int(cfg.model.num_players),
        kinematic_dim=int(cfg.token.kinematic_dim),
        field_length=float(cfg.data.field_length),
        field_width=float(cfg.data.field_width),
    )
    train_dl = DataLoader(train_ds, batch_size=int(cfg.training.batch_size),
                          shuffle=True, num_workers=int(cfg.data.num_workers),
                          pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=int(cfg.training.batch_size),
                        shuffle=False, num_workers=2)

    intent_weights = torch.tensor(cfg.get("intent_class_weights"),
                                  device=device, dtype=torch.float32)
    criterion = IntentTotalLoss(
        intent_weights=intent_weights,
        loss_weights=cfg.get("loss_weights"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.training.epochs))

    grad_clip = float(cfg.training.gradient_clip)
    epochs = 2 if dry_run else int(cfg.training.epochs)
    max_batches = 2 if dry_run else None

    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in enumerate(train_dl):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch)
            losses = criterion(pred, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        metrics = compute_metrics(model, val_dl, device)
        print(f"epoch {epoch:3d} | intent_acc={metrics['intent_accuracy']:.4f} "
              f"| dir_acc@1={metrics['direction_acc_at_1']:.4f} "
              f"| ade={metrics['ade_m']:.3f}m")

        if metrics["intent_accuracy"] > best_acc:
            best_acc = metrics["intent_accuracy"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics},
                       out_dir / "best.pt")

    print(f"Best intent accuracy: {best_acc:.4f}")
    return best_acc
