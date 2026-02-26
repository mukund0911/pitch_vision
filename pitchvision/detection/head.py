"""
YOLO-style detection head.

Simplified single-scale variant inspired by YOLOv1 (Redmon et al., 2016).
Takes one backbone feature map and produces a dense grid of predictions.

Per cell: `num_boxes` boxes each with (tx, ty, tw, th, objectness), plus a
single set of class logits shared across the boxes.
"""

import torch
import torch.nn as nn


class YOLOHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_boxes: int = 2,
                 num_classes: int = 2):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.out_per_cell = num_boxes * 5 + num_classes

        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.conv_out = nn.Conv2d(512, self.out_per_cell, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(features)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.conv_out(x)
        # (B, C, S, S) -> (B, S, S, C)
        return x.permute(0, 2, 3, 1).contiguous()

    def decode(self, raw_preds: torch.Tensor, img_size: int = 416) -> torch.Tensor:
        """
        Turn raw grid logits into absolute-pixel box predictions.

        Returns:
            (B, S, S, num_boxes, 6) — [x1, y1, x2, y2, confidence, class_id]
        """
        B, S, _, _ = raw_preds.shape
        stride = img_size / S
        device = raw_preds.device

        box_slice = raw_preds[..., : self.num_boxes * 5]
        cls_logits = raw_preds[..., self.num_boxes * 5 :]

        box_preds = box_slice.view(B, S, S, self.num_boxes, 5)

        tx = box_preds[..., 0]
        ty = box_preds[..., 1]
        tw = box_preds[..., 2]
        th = box_preds[..., 3]
        to = box_preds[..., 4]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=device, dtype=raw_preds.dtype),
            torch.arange(S, device=device, dtype=raw_preds.dtype),
            indexing="ij",
        )
        grid_x = grid_x.view(1, S, S, 1)
        grid_y = grid_y.view(1, S, S, 1)

        # Pixel-space centers and sizes.
        bx = (tx + grid_x) * stride
        by = (ty + grid_y) * stride
        bw = torch.exp(tw.clamp(max=6.0)) * stride
        bh = torch.exp(th.clamp(max=6.0)) * stride
        obj = torch.sigmoid(to)

        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2

        cls_probs = torch.softmax(cls_logits, dim=-1)
        cls_conf, cls_id = cls_probs.max(dim=-1)
        cls_conf = cls_conf.unsqueeze(-1)
        cls_id = cls_id.unsqueeze(-1).float()

        confidence = obj * cls_conf
        class_id = cls_id.expand_as(obj)

        decoded = torch.stack([x1, y1, x2, y2, confidence, class_id], dim=-1)
        return decoded
