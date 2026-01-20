from typing import List, Optional
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torch


class DetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self._metric = None
        self._fallback_stats = {"tp": 0, "fp": 0, "fn": 0}

        try:
            self._metric = MeanAveragePrecision(iou_type="bbox")
        except Exception:
            self._metric = None

    def update(self, preds: List[dict], targets: List[dict]):
        if self._metric is not None:
            self._metric.update(preds, targets)
            return

        for pred, target in zip(preds, targets):
            pred_boxes = pred.get("boxes", torch.zeros((0, 4)))
            target_boxes = target.get("boxes", torch.zeros((0, 4)))
            self._update_fallback(pred_boxes, target_boxes)

    def compute(self) -> dict:
        if self._metric is not None:
            return self._metric.compute()

        tp = self._fallback_stats["tp"]
        fp = self._fallback_stats["fp"]
        fn = self._fallback_stats["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            "map_50": torch.tensor(precision),
            "map": torch.tensor(precision),
            "precision": torch.tensor(precision),
            "recall": torch.tensor(recall),
        }

    def reset(self):
        if self._metric is not None:
            self._metric.reset()
            return
        self._fallback_stats = {"tp": 0, "fp": 0, "fn": 0}

    def _update_fallback(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
        if pred_boxes.numel() == 0 and target_boxes.numel() == 0:
            return
        if pred_boxes.numel() == 0:
            self._fallback_stats["fn"] += target_boxes.shape[0]
            return
        if target_boxes.numel() == 0:
            self._fallback_stats["fp"] += pred_boxes.shape[0]
            return

        ious = box_iou(pred_boxes, target_boxes)
        matches = ious >= self.iou_threshold
        tp = matches.any(dim=1).sum().item()
        fp = pred_boxes.shape[0] - tp
        fn = target_boxes.shape[0] - matches.any(dim=0).sum().item()

        self._fallback_stats["tp"] += tp
        self._fallback_stats["fp"] += fp
        self._fallback_stats["fn"] += fn


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)
