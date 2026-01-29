import argparse
import json
import os
import time
import torch
from pathlib import Path
from PIL import Image
from torchvision import tv_tensors
from torch.utils.data import DataLoader
from codes.evaluate_detection import evaluate_detection
from codes.metrics import DetectionMetrics
from data.detection_transforms import build_transforms
from data.visdrone_dataset import VisDroneDataset, collate_fn
from models.detectors import build_detector, build_yolo
from utils.class_names import class_names
from utils.config import load_config
from utils.visdrone_io import load_visdrone_annotations
from utils.visualization import draw_detections, draw_targets, side_by_side, tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Inference su immagini VisDrone")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--model", required=True, choices=["retinanet", "faster_rcnn", "yolov11"])
    parser.add_argument("--score-threshold", type=float, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path a una singola immagine")
    group.add_argument("--dataset", help="Path a un dataset dentro ./Test/<nome> con images/annotations/predictions")
    return parser.parse_args()


#Esegue inferenza torchvision su una singola immagine e salva overlay GT vs Pred affiancati
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
    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)
    target = {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
    }
    width, height = image.size
    target["boxes"] = tv_tensors.BoundingBoxes(
        target["boxes"], format="XYXY", canvas_size=(height, width)
    )
    resized_image, resized_target = transforms(image, target)
    resized_pil = tensor_to_pil(
        resized_image, config["data"]["normalize"], apply_unnorm=apply_unnorm
    )

    with torch.no_grad():
        outputs = model([resized_image.to(device)])[0]

    gt_overlay = resized_pil.copy()
    if resized_target["boxes"].numel() > 0:
        gt_overlay = draw_targets(
            gt_overlay,
            resized_target,
            class_names,
            zero_based_labels=False,
            color="green",
        )
    pred_overlay = draw_detections(
        resized_pil.copy(),
        outputs,
        class_names,
        score_threshold,
        zero_based_labels=zero_based,
        color="red",
    )

    return side_by_side(gt_overlay, pred_overlay)

#Esegue inference yolo su una singola immagine e salva overlay GT vs Pred affiancati
def run_on_image_yolo(image_path, ann_path, model, class_map, score_threshold):
    image = Image.open(image_path).convert("RGB")
    boxes, labels = load_visdrone_annotations(ann_path, class_map)
    gt_overlay = image.copy()
    if boxes:
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        gt_overlay = draw_targets(
            gt_overlay,
            target,
            class_names,
            zero_based_labels=False,
            color="green",
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
            pred_overlay = draw_detections(
                pred_overlay,
                outputs,
                class_names,
                score_threshold=score_threshold,
                zero_based_labels=True,
                color="red",
            )

    return side_by_side(gt_overlay, pred_overlay)

#Esegue inferenza per singola immagine, scegliendo la pipeline corretta in base al modello
def render_gt_vs_pred(
    image_path,
    ann_path,
    model,
    transforms,
    config,
    device,
    score_threshold,
    zero_based,
    apply_unnorm,
    class_map,
    model_name,
):
    if model_name == "yolov11":
        return run_on_image_yolo(image_path, ann_path, model, class_map, score_threshold)
    return run_on_image(
        image_path,
        ann_path,
        model,
        transforms,
        config,
        device,
        score_threshold,
        zero_based,
        apply_unnorm=apply_unnorm,
    )

#un detector TorchVision su un dataset e misura il tempo medio di inferenza
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

#Valuta YOLO su dataset iterando immagini una ad una e aggiornando le metriche in formato compatibile
def evaluate_yolo_dataset(
    model, dataset, score_threshold, eval_use_score_threshold
):
    metrics = DetectionMetrics(score_threshold=score_threshold)
    metrics.reset()
    total_images = 0
    total_pred_time = 0.0

    for idx, (_image, target) in enumerate(dataset):
        image_path = dataset.image_files[idx]
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
        if pred_weights:
            if isinstance(pred_weights, str):
                pred_weights = pred_weights.strip()
            if pred_weights.lower() in ("none", ""):
                model_cfg["weights"] = "none"
            elif pred_weights.lower() in ("coco", "default"):
                model_cfg["weights"] = pred_weights
            elif os.path.exists(pred_weights):
                model_cfg["weights"] = pred_weights
            else:
                raise SystemExit(f"Pesi predizione non validi: {pred_weights}")
        model = build_detector(model_cfg, num_classes=config["data"]["num_classes"])
        model.to(device)
        model.eval()

    if args.image:
        image_path = Path(args.image)
        ann_path = image_path.with_suffix(".txt")
        if not ann_path.exists():
            if image_path.parent.name.lower() == "images":
                ann_path = image_path.parent.parent / "annotations" / f"{image_path.stem}.txt"
            else:
                ann_path = image_path.parent / "annotations" / f"{image_path.stem}.txt"
        output_image = render_gt_vs_pred(
            image_path,
            ann_path,
            model,
            transforms,
            config,
            device,
            score_threshold,
            zero_based,
            apply_unnorm=not use_model_internal_preprocessing,
            class_map=class_map,
            model_name=args.model,
        )
        output_path = image_path.with_suffix("").as_posix() + f"_gt_pred_{args.model}.png"
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
        output_image = render_gt_vs_pred(
            image_path,
            ann_path,
            model,
            transforms,
            config,
            device,
            score_threshold,
            zero_based,
            apply_unnorm=not use_model_internal_preprocessing,
            class_map=class_map,
            model_name=args.model,
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
            model, dataset, score_threshold, eval_use_score_threshold
        )
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


if __name__ == "__main__":
    main()
