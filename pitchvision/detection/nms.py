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

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        (N, M) tensor of IoU values in [0, 1].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Pairwise intersection corners via broadcasting (N, 1, 4) vs (1, M, 4)
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
        iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Greedy non-maximum suppression.

    Args:
        boxes: (N, 4) boxes in [x1, y1, x2, y2].
        scores: (N,) confidence scores.
        iou_threshold: boxes with IoU above this value are suppressed.
    Returns:
        (K,) LongTensor of indices to keep.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = compute_iou(boxes[i].unsqueeze(0), boxes[remaining])[0]
        order = remaining[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def multiclass_nms(boxes: torch.Tensor, scores: torch.Tensor,
                   class_ids: torch.Tensor, iou_threshold: float = 0.5,
                   conf_threshold: float = 0.3) -> tuple:
    """
    Per-class NMS so that e.g. a person box doesn't suppress an overlapping ball.

    Args:
        boxes: (N, 4)
        scores: (N,)
        class_ids: (N,) class label per box
    Returns:
        (kept_boxes, kept_scores, kept_class_ids)
    """
    # Confidence threshold first — cheaper than running NMS on low-conf noise.
    conf_mask = scores >= conf_threshold
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    class_ids = class_ids[conf_mask]

    if boxes.numel() == 0:
        return (torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long))

    kept_boxes, kept_scores, kept_classes = [], [], []
    for c in class_ids.unique():
        mask = class_ids == c
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        keep_idx = nms(cls_boxes, cls_scores, iou_threshold)
        kept_boxes.append(cls_boxes[keep_idx])
        kept_scores.append(cls_scores[keep_idx])
        kept_classes.append(torch.full((keep_idx.numel(),), c.item(), dtype=torch.long))

    return (torch.cat(kept_boxes, dim=0),
            torch.cat(kept_scores, dim=0),
            torch.cat(kept_classes, dim=0))
