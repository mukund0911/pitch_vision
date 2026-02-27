"""
Non-Maximum Suppression.

Removes duplicate/overlapping detections by keeping only the highest confidence
box among a cluster of overlapping predictions.

Reference: YOLO (Redmon 2016), Section 2.2.
"""

import torch


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of boxes in [x1, y1, x2, y2] format.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.3) -> torch.Tensor:
    """
    Greedy NMS with a confidence threshold.

    Returns the indices of the boxes to keep.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    keep_mask = scores >= conf_threshold
    if not keep_mask.any():
        return torch.empty(0, dtype=torch.long)

    surviving = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    boxes = boxes[surviving]
    scores = scores[surviving]

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(surviving[i].item())
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = compute_iou(boxes[i].unsqueeze(0), boxes[remaining])[0]
        order = remaining[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)
