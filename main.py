import argparse
import json
import os
import time

import torch

from codes.train_detection import train_detection
from codes.train_yolo import train_yolo
from data.data_loader import load_data
from models.detectors import build_detector, build_yolo
from utils.checkpoints import load_checkpoint
from utils.clear_console import clear_console
from utils.time_manager import get_current_time


def parse_args():
    parser = argparse.ArgumentParser(description="VisDrone2019 Object Detection Trainer")
    parser.add_argument("--config", default="config/config.json", help="Path to config JSON")
    parser.add_argument("--model", choices=["yolov11", "retinanet", "faster_rcnn"])
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)

    class_map = config["data"].get("class_map", {})
    config["data"]["class_map"] = {int(k): int(v) for k, v in class_map.items()}
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    clear_console()
    print(f"[ Avvio training alle ore {get_current_time()} ]")

    torch.manual_seed(config["experiment"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["experiment"]["seed"])

    if not args.model:
        raise SystemExit("Specifica --model.")

    run_single_model(args.model, config, args.resume)


def run_single_model(model_name, config, resume):
    if model_name == "yolov11":
        yolo_model = build_yolo(config["models"]["yolov11"])
        train_yolo(config, yolo_model, resume=resume)
        return

    train_loader, val_loader = load_data(config)

    model_cfg = config["models"][model_name]
    model = build_detector(model_cfg, num_classes=config["data"]["num_classes"])

    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    optimizer_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg["lr"],
        weight_decay=optimizer_cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, config)

    runs_dir = config["training"]["runs_dir"]
    model_run_dir = os.path.join(runs_dir, model_name)
    os.makedirs(model_run_dir, exist_ok=True)

    if resume:
        checkpoint_path = config["training"].get("resume_checkpoint_path")
        if not checkpoint_path:
            raise SystemExit("Imposta training.resume_checkpoint_path per riprendere il training.")
        if not os.path.exists(checkpoint_path):
            raise SystemExit(f"Checkpoint non trovato: {checkpoint_path}")
        run_dir = os.path.dirname(checkpoint_path)
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scheduler, device=device)
        print(f"Ripresa addestramento da epoca {start_epoch + 1}")
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{model_name}"
        run_dir = os.path.join(model_run_dir, run_name)
        start_epoch = 0

    label_offset = -1 if model_name == "retinanet" else 0
    train_detection(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device,
        start_epoch=start_epoch,
        label_offset=label_offset,
        run_dir=run_dir,
    )


def build_scheduler(optimizer, config):
    training_cfg = config["training"]
    optimizer_cfg = training_cfg["optimizer"]
    total_epochs = training_cfg["epochs"]
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
    warmup_start_factor = float(training_cfg.get("warmup_start_factor", 0.1))

    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=optimizer_cfg["min_lr"]
    )

    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_epochs
    )
    if warmup_epochs >= total_epochs:
        return warmup

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


if __name__ == "__main__":
    main()
