import argparse
import json
import os
import time
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import tv_tensors
from torch.utils.data import DataLoader
from codes.evaluate_detection import evaluate_detection
from codes.metrics import DetectionMetrics
from data.detection_transforms import build_transforms
from data.visdrone_dataset import VisDroneDataset, collate_fn
from models.detectors import build_detector, build_yolo
from utils.checkpoints import load_checkpoint
from utils.class_names import class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Inference su immagini VisDrone")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--model", required=True, choices=["retinanet", "faster_rcnn", "yolov11"])
    parser.add_argument("--score-threshold", type=float, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path a una singola immagine")
    group.add_argument("--dataset", help="Path a un dataset dentro ./Test/<nome> con images/annotations/predictions")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    class_map = config["data"].get("class_map", {})
    config["data"]["class_map"] = {int(k): int(v) for k, v in class_map.items()}
    return config


def draw_boxes(
    image,
    outputs,
    score_threshold,
    zero_based_labels=False,
    color="red",
    show_scores=True,
):
    draw = ImageDraw.Draw(image)
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs.get("scores")
    scores = scores.cpu().numpy() if scores is not None else None
    labels = outputs["labels"].cpu().numpy()

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        score = scores[idx] if scores is not None else None
        if score is not None and score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if zero_based_labels:
            class_label = class_names[label] if 0 <= label < len(class_names) else f"id-{label}"
        else:
            class_label = class_names[label - 1] if 0 < label <= len(class_names) else f"id-{label}"
        if show_scores and score is not None:
            text = f"{class_label} {score:.2f}"
        else:
            text = f"{class_label}"
        draw.text((x1, y1), text, fill=color)

    return image


def load_visdrone_annotations(ann_path, class_map):
    boxes = []
    labels = []
    if not os.path.exists(ann_path):
        return boxes, labels

    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h, score, category = parts[:6]
            category = int(category)
            if category <= 0 or category not in class_map:
                continue
            mapped = class_map[category]
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            boxes.append([x, y, x + w, y + h])
            labels.append(mapped)

    return boxes, labels


def prepare_image(image, target, transforms, normalize_cfg, apply_unnorm):
    resized_image, resized_target = transforms(image, target)
    resized_pil = tensor_to_pil(resized_image, normalize_cfg, apply_unnorm=apply_unnorm)
    return resized_image, resized_target, resized_pil


def side_by_side(left, right):
    width = left.width + right.width
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def run_on_image(
    image_path,
    ann_path,
    model,
    transforms,
    config,
    device,
    score_threshold,
    zero_based,
    apply_unnorm,
):
    image = Image.open(image_path).convert("RGB")
    class_map = config["data"]["class_map"]
    boxes, labels = load_visdrone_annotations(ann_path, class_map)
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    width, height = image.size
    target["boxes"] = tv_tensors.BoundingBoxes(
        target["boxes"], format="XYXY", canvas_size=(height, width)
    )
    resized_image, resized_target, resized_pil = prepare_image(
        image, target, transforms, config["data"]["normalize"], apply_unnorm
    )

    with torch.no_grad():
        outputs = model([resized_image.to(device)])[0]

    gt_overlay = resized_pil.copy()
    if resized_target["boxes"].numel() > 0:
        gt_overlay = draw_boxes(
            gt_overlay,
            resized_target,
            score_threshold=0.0,
            zero_based_labels=False,
            color="green",
            show_scores=False,
        )
    pred_overlay = draw_boxes(
        resized_pil.copy(),
        outputs,
        score_threshold,
        zero_based_labels=zero_based,
        color="red",
        show_scores=True,
    )

    return side_by_side(gt_overlay, pred_overlay)


def run_on_image_yolo(image_path, ann_path, model, class_map, score_threshold):
    image = Image.open(image_path).convert("RGB")
    boxes, labels = load_visdrone_annotations(ann_path, class_map)
    gt_overlay = image.copy()
    if boxes:
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        gt_overlay = draw_boxes(
            gt_overlay,
            target,
            score_threshold=0.0,
            zero_based_labels=False,
            color="green",
            show_scores=False,
        )

    results = model.predict(
        source=str(image_path),
        conf=score_threshold,
        verbose=False,
        save=False,
    )
    pred_overlay = image.copy()
    if results:
        res = results[0]
        if res.boxes is not None and res.boxes.xyxy is not None:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
            labels_pred = res.boxes.cls.cpu().numpy().astype(int)
            outputs = {
                "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
                "scores": torch.tensor(scores, dtype=torch.float32) if scores is not None else None,
                "labels": torch.tensor(labels_pred, dtype=torch.int64),
            }
            pred_overlay = draw_boxes(
                pred_overlay,
                outputs,
                score_threshold=score_threshold,
                zero_based_labels=True,
                color="red",
                show_scores=True,
            )

    return side_by_side(gt_overlay, pred_overlay)


def evaluate_detector_dataset(
    model,
    dataset,
    device,
    label_offset,
    batch_size,
    num_workers,
    score_threshold,
    track_inference_time,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    metrics = DetectionMetrics(score_threshold=score_threshold)
    val_loss, metric_values, avg_pred_time = evaluate_detection(
        model,
        loader,
        device,
        metrics,
        label_offset=label_offset,
        track_inference_time=track_inference_time,
        compute_loss=False,
    )
    return val_loss, metric_values, avg_pred_time


def evaluate_yolo_dataset(
    model, dataset, device, class_map, score_threshold, eval_use_score_threshold
):
    metrics = DetectionMetrics(score_threshold=score_threshold)
    metrics.reset()
    total_images = 0
    total_pred_time = 0.0

    for image, target in dataset:
        image_path = dataset.image_files[total_images]
        image_path = os.path.join(dataset.images_dir, image_path)
        start = time.perf_counter()
        conf = score_threshold if eval_use_score_threshold else 0.0
        results = model.predict(source=image_path, conf=conf, verbose=False, save=False)
        total_pred_time += time.perf_counter() - start

        preds = []
        if results:
            res = results[0]
            if res.boxes is not None and res.boxes.xyxy is not None:
                boxes_xyxy = torch.as_tensor(res.boxes.xyxy.cpu().numpy(), dtype=torch.float32)
                scores = torch.as_tensor(res.boxes.conf.cpu().numpy(), dtype=torch.float32)
                labels = torch.as_tensor(res.boxes.cls.cpu().numpy().astype(int), dtype=torch.int64) + 1
                preds = [{"boxes": boxes_xyxy, "scores": scores, "labels": labels}]
        if not preds:
            preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64)}]

        metrics.update(preds, [target])
        total_images += 1

    metric_values = metrics.compute()
    avg_pred_time = total_pred_time / max(total_images, 1)
    return metric_values, avg_pred_time


def main():
    args = parse_args()
    config = load_config(args.config)
    prediction_cfg = config.get("prediction", {})
    prediction_weights = prediction_cfg.get("model_weights", {})
    if args.score_threshold is None:
        score_threshold = float(
            prediction_cfg.get(
                "score_threshold",
                config.get("evaluation", {}).get("score_threshold", 0.4),
            )
        )
    else:
        score_threshold = args.score_threshold
    eval_use_score_threshold = bool(prediction_cfg.get("eval_use_score_threshold", True))

    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    use_model_internal_preprocessing = args.model in {"retinanet", "faster_rcnn"}
    transforms = build_transforms(
        config,
        train=False,
        use_model_internal_preprocessing=use_model_internal_preprocessing,
    )
    zero_based = args.model == "retinanet"
    class_map = config["data"]["class_map"]

    if args.model == "yolov11":
        yolo_cfg = dict(config["models"]["yolov11"])
        yolo_weights = prediction_weights.get("yolov11")
        if yolo_weights:
            yolo_cfg["weights"] = yolo_weights
        model = build_yolo(yolo_cfg)
    else:
        model_cfg = dict(config["models"][args.model])
        pred_weights = prediction_weights.get(args.model)
        checkpoint_path = None
        if pred_weights:
            if isinstance(pred_weights, str):
                pred_weights = pred_weights.strip()
            if pred_weights.lower() in ("none", ""):
                model_cfg["weights"] = "none"
            elif pred_weights.lower() in ("coco", "default"):
                model_cfg["weights"] = pred_weights
            elif os.path.exists(pred_weights):
                model_cfg["weights"] = "none"
                checkpoint_path = pred_weights
            else:
                raise SystemExit(f"Pesi predizione non validi: {pred_weights}")
        model = build_detector(model_cfg, num_classes=config["data"]["num_classes"])
        if checkpoint_path:
            load_checkpoint(model, None, checkpoint_path, device=device)
        model.to(device)
        model.eval()

    if args.image:
        image_path = Path(args.image)
        ann_path = image_path.with_suffix(".txt")
        if args.model == "yolov11":
            output_image = run_on_image_yolo(
                image_path, ann_path, model, class_map, score_threshold
            )
        else:
            output_image = run_on_image(
                image_path,
                ann_path,
                model,
                transforms,
                config,
                device,
                score_threshold,
                zero_based,
                apply_unnorm=not use_model_internal_preprocessing,
            )
        output_path = image_path.with_suffix("").as_posix() + "_gt_pred.png"
        output_image.save(output_path)
        print(f"Output salvato in {output_path}")
        return

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    if not annotations_dir.exists():
        raise SystemExit(f"Cartella annotations mancante: {annotations_dir}")
    base_predictions_dir = dataset_dir / "predictions"
    base_predictions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"predict_{args.model}_{timestamp}"
    predictions_dir = base_predictions_dir / run_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir = predictions_dir / "side_by_side"
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_path in image_files:
        ann_path = annotations_dir / f"{image_path.stem}.txt"
        if args.model == "yolov11":
            output_image = run_on_image_yolo(
                image_path, ann_path, model, class_map, score_threshold
            )
        else:
            output_image = run_on_image(
                image_path,
                ann_path,
                model,
                transforms,
                config,
                device,
                score_threshold,
                zero_based,
                apply_unnorm=not use_model_internal_preprocessing,
            )
        output_path = side_by_side_dir / f"{image_path.stem}_gt_pred.png"
        output_image.save(output_path)

    metrics_path = predictions_dir / "metrics.json"
    if args.model == "yolov11":
        dataset = VisDroneDataset(
            str(images_dir),
            str(annotations_dir),
            transforms=None,
            class_map=class_map,
            valid_categories=config["data"].get("valid_categories"),
        )
        metric_values, avg_pred_time = evaluate_yolo_dataset(
            model, dataset, device, class_map, score_threshold, eval_use_score_threshold
        )
        metrics_out = {
            "map_50": float(metric_values.get("map_50", 0.0)),
            "map_50_95": float(metric_values.get("map", 0.0)),
            "precision": float(metric_values.get("precision", 0.0)),
            "recall": float(metric_values.get("recall", 0.0)),
            "avg_inference_time_sec": avg_pred_time,
        }
    else:
        dataset = VisDroneDataset(
            str(images_dir),
            str(annotations_dir),
            transforms=build_transforms(
                config,
                train=False,
                use_model_internal_preprocessing=use_model_internal_preprocessing,
            ),
            class_map=class_map,
            valid_categories=config["data"].get("valid_categories"),
        )
        label_offset = -1 if args.model == "retinanet" else 0
        val_loss, metric_values, avg_pred_time = evaluate_detector_dataset(
            model,
            dataset,
            device,
            label_offset,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            score_threshold=score_threshold,
            track_inference_time=True,
        )
        metrics_out = {
            "map_50": float(metric_values.get("map_50", 0.0)),
            "map_50_95": float(metric_values.get("map", 0.0)),
            "precision": float(metric_values.get("precision", 0.0)),
            "recall": float(metric_values.get("recall", 0.0)),
            "avg_inference_time_sec": avg_pred_time,
        }
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Output salvato in {predictions_dir}")


def tensor_to_pil(tensor_image, normalize_cfg, apply_unnorm=True):
    image = tensor_image.cpu()
    if apply_unnorm:
        mean = torch.tensor(normalize_cfg["mean"]).view(3, 1, 1)
        std = torch.tensor(normalize_cfg["std"]).view(3, 1, 1)
        image = image * std + mean
    image = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)


if __name__ == "__main__":
    main()
