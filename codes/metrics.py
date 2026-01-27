from typing import List
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

#Gestisce mAP/precision/recall con TorchMetrics
class DetectionMetrics:

    def __init__(self, iou_threshold=0.5, score_threshold=0.4):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self._pr_stats = {"tp": 0, "fp": 0, "fn": 0}
        self._metric = MeanAveragePrecision(iou_type="bbox")

    #Aggiorna la metrica principale e le statistiche PR
    def update(self, preds: List[dict], targets: List[dict]):
        self._metric.update(preds, targets)

        for pred, target in zip(preds, targets):
            pred_boxes = pred.get("boxes", torch.zeros((0, 4)))
            target_boxes = target.get("boxes", torch.zeros((0, 4)))
            pred_scores = pred.get("scores", torch.zeros((pred_boxes.shape[0],)))
            self._update_precision_recall(pred_boxes, pred_scores, target_boxes)

    #Calcola le metriche finali da TorchMetrics e aggiunge precision/recall
    def compute(self) -> dict:
        metric_vals = self._metric.compute()
        pr_tp = self._pr_stats["tp"]
        pr_fp = self._pr_stats["fp"]
        pr_fn = self._pr_stats["fn"]
        pr_precision = pr_tp / max(pr_tp + pr_fp, 1)
        pr_recall = pr_tp / max(pr_tp + pr_fn, 1)
        metric_vals["precision"] = torch.tensor(pr_precision)
        metric_vals["recall"] = torch.tensor(pr_recall)
        return metric_vals

    #Azzera lo stato interno delle metriche.
    def reset(self):
        self._metric.reset()
        self._pr_stats = {"tp": 0, "fp": 0, "fn": 0}

    #Precision/recall calcolate su predizioni filtrate per score
    def _update_precision_recall(
        self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, target_boxes: torch.Tensor
    ):
        #Se non c'e nulla da valutare, non aggiorna nulla
        if pred_boxes.numel() == 0 and target_boxes.numel() == 0:
            return

        #Se non ci sono predizioni ma ci sono target, sono tutti falsi negativi
        if pred_boxes.numel() == 0:
            self._pr_stats["fn"] += target_boxes.shape[0]
            return

        #Tiene solo le predizioni con confidenza sufficiente
        keep = pred_scores >= self.score_threshold
        pred_boxes = pred_boxes[keep]

        #Se dopo il filtro non ci sono predizioni valide, tutti i target sono FN
        if pred_boxes.numel() == 0:
            self._pr_stats["fn"] += target_boxes.shape[0]
            return

        #Se non ci sono target ma ci sono predizioni, sono tutti falsi positivi
        if target_boxes.numel() == 0:
            self._pr_stats["fp"] += pred_boxes.shape[0]
            return

        ious = self._box_iou(pred_boxes, target_boxes)
        matched_targets = set()
        tp = 0

        #Per ogni predizione trova il target con IoU massimo
        #Se supera la soglia e non e' gia' stato abbinato, e' un vero positivo
        #Aggiorna TP, FP, FN di conseguenza
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

    #Calcola la matrice IoU tra due insiemi di box
    #clamp() evita aree negative se i box è malformato
    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter
        return inter / union.clamp(min=1e-6)
