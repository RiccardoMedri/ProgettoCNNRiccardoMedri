import os
from typing import Dict, Optional


def _build_anchor_generator(anchor_cfg: Optional[Dict]):
    if not anchor_cfg:
        return None

    from torchvision.models.detection.anchor_utils import AnchorGenerator

    sizes = anchor_cfg.get("sizes")
    if not sizes:
        return None
    if all(isinstance(s, (int, float)) for s in sizes):
        sizes = tuple((int(s),) for s in sizes)
    else:
        sizes = tuple(tuple(int(v) for v in s) for s in sizes)

    aspect_ratios = anchor_cfg.get("aspect_ratios", (0.5, 1.0, 2.0))
    if all(isinstance(ar, (int, float)) for ar in aspect_ratios):
        aspect_ratios = tuple(float(ar) for ar in aspect_ratios)
        aspect_ratios = tuple(aspect_ratios for _ in range(len(sizes)))
    else:
        aspect_ratios = tuple(tuple(float(ar) for ar in ars) for ars in aspect_ratios)

    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def build_detector(model_cfg: Dict, num_classes: int):
    model_type = model_cfg["type"].lower()

    if model_type == "faster_rcnn":
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            fasterrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        anchor_generator = _build_anchor_generator(model_cfg.get("anchors"))
        backbone = model_cfg.get("backbone", "resnet50_fpn_v2").lower()
        if backbone == "resnet50_fpn_v2":
            model = fasterrcnn_resnet50_fpn_v2(
                weights="DEFAULT",
                trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
                rpn_anchor_generator=anchor_generator,
            )
        else:
            model = fasterrcnn_resnet50_fpn(
                weights="DEFAULT",
                trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
                rpn_anchor_generator=anchor_generator,
            )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        effective_classes = num_classes + 1
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, effective_classes)
        return model

    if model_type == "retinanet":
        from torchvision.models.detection import (
            retinanet_resnet50_fpn_v2,
            RetinaNet_ResNet50_FPN_V2_Weights,
        )
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        from torchvision.models import ResNet50_Weights
        import torch

        effective_classes = num_classes
        weights_cfg = model_cfg.get("weights", "coco")
        if isinstance(weights_cfg, str):
            weights_cfg = weights_cfg.strip()

        anchor_generator = _build_anchor_generator(model_cfg.get("anchors"))

        if weights_cfg in (None, "", "none"):
            model = retinanet_resnet50_fpn_v2(
                weights=None,
                weights_backbone=ResNet50_Weights.DEFAULT,
                num_classes=effective_classes,
                trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
                anchor_generator=anchor_generator,
            )
            return model

        if isinstance(weights_cfg, str) and weights_cfg.lower() in ("coco", "default"):
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            model = retinanet_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=None,
                trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
                anchor_generator=anchor_generator,
            )
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels, num_anchors, effective_classes
            )
            return model

        if isinstance(weights_cfg, str) and os.path.exists(weights_cfg):
            model = retinanet_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=effective_classes,
                trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
                anchor_generator=anchor_generator,
            )
            checkpoint = torch.load(weights_cfg, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            return model

        raise ValueError(f"Pesi RetinaNet non validi: {weights_cfg}")

    raise ValueError(f"Tipo modello non supportato: {model_type}")


def build_yolo(model_cfg: Dict):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Per YOLOv11 installa ultralytics.") from exc

    weights = model_cfg.get("weights", "yolov11s.pt")
    return YOLO(weights)
