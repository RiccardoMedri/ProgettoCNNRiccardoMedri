from typing import List, Optional
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torch


class DetectionMetrics:
    def __init__(self, iou_threshold=0.5, score_threshold=0.4):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self._metric = None
        self._fallback_stats = {"tp": 0, "fp": 0, "fn": 0}
        self._pr_stats = {"tp": 0, "fp": 0, "fn": 0}

        try:
            self._metric = MeanAveragePrecision(iou_type="bbox")
        except Exception:
            self._metric = None

    def update(self, preds: List[dict], targets: List[dict]):
        if self._metric is not None:
            self._metric.update(preds, targets)

        for pred, target in zip(preds, targets):
            pred_boxes = pred.get("boxes", torch.zeros((0, 4)))
            target_boxes = target.get("boxes", torch.zeros((0, 4)))
            pred_scores = pred.get("scores", torch.zeros((pred_boxes.shape[0],)))
            self._update_fallback(pred_boxes, target_boxes)
            self._update_precision_recall(pred_boxes, pred_scores, target_boxes)

    def compute(self) -> dict:
        if self._metric is not None:
            metric_vals = self._metric.compute()
            pr_tp = self._pr_stats["tp"]
            pr_fp = self._pr_stats["fp"]
            pr_fn = self._pr_stats["fn"]
            pr_precision = pr_tp / max(pr_tp + pr_fp, 1)
            pr_recall = pr_tp / max(pr_tp + pr_fn, 1)
            metric_vals["precision"] = torch.tensor(pr_precision)
            metric_vals["recall"] = torch.tensor(pr_recall)
            return metric_vals

        tp = self._fallback_stats["tp"]
        fp = self._fallback_stats["fp"]
        fn = self._fallback_stats["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        pr_tp = self._pr_stats["tp"]
        pr_fp = self._pr_stats["fp"]
        pr_fn = self._pr_stats["fn"]
        pr_precision = pr_tp / max(pr_tp + pr_fp, 1)
        pr_recall = pr_tp / max(pr_tp + pr_fn, 1)
        return {
            "map_50": torch.tensor(precision),
            "map": torch.tensor(precision),
            "precision": torch.tensor(pr_precision),
            "recall": torch.tensor(pr_recall),
        }

    def reset(self):
        if self._metric is not None:
            self._metric.reset()
        self._fallback_stats = {"tp": 0, "fp": 0, "fn": 0}
        self._pr_stats = {"tp": 0, "fp": 0, "fn": 0}

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

    def _update_precision_recall(
        self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, target_boxes: torch.Tensor
    ):
        if pred_boxes.numel() == 0 and target_boxes.numel() == 0:
            return
        if pred_boxes.numel() == 0:
            self._pr_stats["fn"] += target_boxes.shape[0]
            return

        keep = pred_scores >= self.score_threshold
        pred_boxes = pred_boxes[keep]
        if pred_boxes.numel() == 0:
            self._pr_stats["fn"] += target_boxes.shape[0]
            return

        if target_boxes.numel() == 0:
            self._pr_stats["fp"] += pred_boxes.shape[0]
            return

        ious = box_iou(pred_boxes, target_boxes)
        matched_targets = set()
        tp = 0
        for pred_idx in range(ious.shape[0]):
            best_iou, best_target = torch.max(ious[pred_idx], dim=0)
            if best_iou >= self.iou_threshold and best_target.item() not in matched_targets:
                tp += 1
                matched_targets.add(best_target.item())

        fp = pred_boxes.shape[0] - tp
        fn = target_boxes.shape[0] - len(matched_targets)

        self._pr_stats["tp"] += tp
        self._pr_stats["fp"] += fp
        self._pr_stats["fn"] += fn

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)
