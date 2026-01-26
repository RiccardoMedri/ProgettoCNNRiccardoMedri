import os
from typing import Dict
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    fasterrcnn_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch


def _apply_transform_cfg(model, model_cfg: Dict):
    transform_cfg = model_cfg.get("transform")
    if not transform_cfg:
        return
    min_size = transform_cfg.get("min_size")
    max_size = transform_cfg.get("max_size")
    if min_size is not None:
        if isinstance(min_size, (list, tuple)):
            model.transform.min_size = list(min_size)
        else:
            model.transform.min_size = [int(min_size)]
    if max_size is not None:
        model.transform.max_size = int(max_size)


def build_detector(model_cfg: Dict, num_classes: int):
    model_type = model_cfg["type"].lower()

    if model_type == "faster_rcnn":
        weights_cfg = model_cfg.get("weights", "default")
        if isinstance(weights_cfg, str):
            weights_cfg = weights_cfg.strip()

        is_none = weights_cfg in (None, "", "none")
        is_coco = isinstance(weights_cfg, str) and weights_cfg.lower() in ("coco", "default")
        is_path = isinstance(weights_cfg, str) and os.path.exists(weights_cfg)
        if not (is_none or is_coco or is_path):
            raise ValueError(f"Pesi Faster R-CNN non validi: {weights_cfg}")

        trainable_backbone_layers = model_cfg.get("trainable_backbone_layers", 3)

        if is_none:
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=ResNet50_Weights.DEFAULT,
                trainable_backbone_layers=trainable_backbone_layers,
            )
        elif is_coco:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=None,
                trainable_backbone_layers=trainable_backbone_layers,
            )
        else:
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=ResNet50_Weights.DEFAULT,
                trainable_backbone_layers=trainable_backbone_layers,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            effective_classes = num_classes + 1
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, effective_classes)
            checkpoint = torch.load(weights_cfg, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            _apply_transform_cfg(model, model_cfg)
            return model

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        effective_classes = num_classes + 1
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, effective_classes)
        _apply_transform_cfg(model, model_cfg)
        return model

    if model_type == "retinanet":
        effective_classes = num_classes
        weights_cfg = model_cfg.get("weights", "coco")
        if isinstance(weights_cfg, str):
            weights_cfg = weights_cfg.strip()

        is_none = weights_cfg in (None, "", "none")
        is_coco = isinstance(weights_cfg, str) and weights_cfg.lower() in ("coco", "default")
        is_path = isinstance(weights_cfg, str) and os.path.exists(weights_cfg)
        if not (is_none or is_coco or is_path):
            raise ValueError(f"Pesi RetinaNet non validi: {weights_cfg}")

        trainable_backbone_layers = model_cfg.get("trainable_backbone_layers", 3)

        if is_none:
            model = retinanet_resnet50_fpn_v2(
                weights=None,
                weights_backbone=ResNet50_Weights.DEFAULT,
                num_classes=effective_classes,
                trainable_backbone_layers=trainable_backbone_layers,
            )
            _apply_transform_cfg(model, model_cfg)
            return model

        if is_coco:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            model = retinanet_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=None,
                trainable_backbone_layers=trainable_backbone_layers,
            )
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels, num_anchors, effective_classes
            )
            _apply_transform_cfg(model, model_cfg)
            return model

        if is_path:
            model = retinanet_resnet50_fpn_v2(
                weights=None,
                weights_backbone=ResNet50_Weights.DEFAULT,
                num_classes=effective_classes,
                trainable_backbone_layers=trainable_backbone_layers,
            )
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels, num_anchors, effective_classes
            )
            checkpoint = torch.load(weights_cfg, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            _apply_transform_cfg(model, model_cfg)
            return model

    raise ValueError(f"Tipo modello non supportato: {model_type}")


def build_yolo(model_cfg: Dict):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Per YOLOv11 installa ultralytics.") from exc

    weights = model_cfg.get("weights", "yolov11s.pt")
    return YOLO(weights)
