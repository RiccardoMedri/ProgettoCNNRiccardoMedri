import os
import time

import torch

from data.visdrone_to_yolo import convert_visdrone_to_yolo, write_yolo_yaml
from utils.class_names import class_names


def train_yolo(config, model, resume=False):
    data_cfg = config["data"]["yolo"]
    dataset_root = data_cfg["dataset_root"]
    dataset_root_abs = os.path.abspath(dataset_root)
    train_images = os.path.join(dataset_root, "train", "images")
    train_ann = os.path.join(dataset_root, "train", "annotations")
    train_labels = os.path.join(dataset_root, "train", "labels")
    val_images = os.path.join(dataset_root, "val", "images")
    val_ann = os.path.join(dataset_root, "val", "annotations")
    val_labels = os.path.join(dataset_root, "val", "labels")

    if data_cfg.get("auto_convert", True):
        convert_visdrone_to_yolo(train_images, train_ann, train_labels, config["data"]["class_map"])
        convert_visdrone_to_yolo(val_images, val_ann, val_labels, config["data"]["class_map"])

    yaml_path = data_cfg.get("yaml_path", os.path.join(dataset_root, "visdrone.yaml"))
    yaml_path_abs = os.path.abspath(yaml_path)
    rewrite_yaml = True
    if os.path.exists(yaml_path_abs):
        with open(yaml_path_abs, "r") as f:
            for line in f:
                if line.strip().startswith("path:"):
                    current = line.split(":", 1)[1].strip()
                    rewrite_yaml = (
                        not os.path.isabs(current)
                        or os.path.abspath(current) != dataset_root_abs
                    )
                    break
    if rewrite_yaml:
        write_yolo_yaml(
            yaml_path_abs,
            dataset_root_abs,
            {i + 1: name for i, name in enumerate(class_names)},
        )

    train_cfg = config["training"]
    runs_dir = os.path.join(train_cfg["runs_dir"], "yolov11")
    os.makedirs(runs_dir, exist_ok=True)

    if resume:
        ckpt = train_cfg.get("resume_checkpoint_path")
        if not ckpt:
            raise SystemExit("Imposta training.resume_checkpoint_path per riprendere il training.")
        if not os.path.exists(ckpt):
            raise SystemExit(f"Checkpoint YOLO non trovato: {ckpt}")
        model.train(resume=ckpt)
        return

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_yolov11"
    run_dir = os.path.join(runs_dir, run_name)

    model.train(
        data=yaml_path_abs,
        epochs=train_cfg["epochs"],
        patience=train_cfg.get("patience"),
        imgsz=config["data"]["image_size"],
        batch=config["data"]["batch_size"],
        lr0=train_cfg["optimizer"]["lr"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
        warmup_epochs=train_cfg.get("warmup_epochs"),
        device=config["experiment"]["device"] if torch.cuda.is_available() else "cpu",
        project=runs_dir,
        name=run_name,
    )
