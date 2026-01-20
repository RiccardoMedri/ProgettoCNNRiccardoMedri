from typing import Dict


def build_detector(model_cfg: Dict, num_classes: int):
    model_type = model_cfg["type"].lower()

    if model_type == "faster_rcnn":
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            fasterrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        backbone = model_cfg.get("backbone", "resnet50_fpn_v2").lower()
        if backbone == "resnet50_fpn_v2":
            model = fasterrcnn_resnet50_fpn_v2(
                weights="DEFAULT",
                trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
            )
        else:
            model = fasterrcnn_resnet50_fpn(
                weights="DEFAULT",
                trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
            )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        effective_classes = num_classes + 1
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, effective_classes)
        return model

    if model_type == "retinanet":
        from torchvision.models.detection import retinanet_resnet50_fpn_v2
        from torchvision.models import ResNet50_Weights

        effective_classes = num_classes

        model = retinanet_resnet50_fpn_v2(
            weights=None,  # <- don't load COCO head (otherwise num_classes must be 91)
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=effective_classes,
            trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
        )
        return model

    raise ValueError(f"Tipo modello non supportato: {model_type}")


def build_yolo(model_cfg: Dict):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Per YOLOv11 installa ultralytics.") from exc

    weights = model_cfg.get("weights", "yolov11s.pt")
    return YOLO(weights)
