"""
Full object detector: ResNet-18 backbone + YOLO head + NMS.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from .backbone import ResNet18
from .head import YOLOHead
from .nms import nms


class Detector(nn.Module):
    PERSON_CLASS = 0
    BALL_CLASS = 1

    def __init__(self, num_classes: int = 2, num_boxes: int = 2,
                 img_size: int = 416, conf_threshold: float = 0.3,
                 nms_threshold: float = 0.5):
        super().__init__()
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.num_classes = num_classes

        self.backbone = ResNet18()
        self.head = YOLOHead(in_channels=256, num_boxes=num_boxes,
                             num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats["feat16"])

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> dict:
        orig_h, orig_w = frame.shape[:2]
        device = next(self.parameters()).device

        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        raw = self.forward(img)
        decoded = self.head.decode(raw, img_size=self.img_size)

        flat = decoded.view(-1, 6)
        boxes = flat[:, :4].cpu()
        scores = flat[:, 4].cpu()
        class_ids = flat[:, 5].long().cpu()

        keep = nms(boxes, scores, iou_threshold=self.nms_threshold,
                   conf_threshold=self.conf_threshold)
        kept_boxes = boxes[keep]
        kept_scores = scores[keep]
        kept_classes = class_ids[keep]

        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size
        kept_boxes[:, [0, 2]] *= scale_x
        kept_boxes[:, [1, 3]] *= scale_y
        kept_boxes[:, [0, 2]] = kept_boxes[:, [0, 2]].clamp(0, orig_w - 1)
        kept_boxes[:, [1, 3]] = kept_boxes[:, [1, 3]].clamp(0, orig_h - 1)

        players, ball = [], None
        for box, score, cid in zip(kept_boxes.tolist(), kept_scores.tolist(),
                                   kept_classes.tolist()):
            entry = box + [score]
            if cid == self.PERSON_CLASS:
                players.append(entry)
            elif cid == self.BALL_CLASS:
                if ball is None or score > ball[4]:
                    ball = entry

        return {"players": players, "ball": ball}
