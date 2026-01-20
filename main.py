import argparse
import json

import torch

from codes.train_detection import train_detection
from codes.train_yolo import train_yolo
from data.data_loader import load_data
from models.detectors import build_detector, build_yolo
from utils.checkpoints import load_checkpoint
from utils.clear_console import clear_console
from utils.config_utils import deep_merge
from utils.time_manager import get_current_time


def parse_args():
    parser = argparse.ArgumentParser(description="VisDrone2019 Object Detection Trainer")
    parser.add_argument("--config", default="config/config.json", help="Path to config JSON")
    parser.add_argument("--model", choices=["yolov11", "retinanet", "faster_rcnn"])
    parser.add_argument("--experiments", default="config/experiments.json", help="Path to experiments JSON")
    parser.add_argument("--run", help="Run a named experiment from experiments.json or 'all'")
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

    if args.run:
        run_experiments(args, config)
        return

    if not args.model:
        raise SystemExit("Specifica --model oppure --run.")

    run_single_model(args.model, config, args.resume)


def run_experiments(args, base_config):
    with open(args.experiments, "r") as f:
        experiments = json.load(f)

    if args.run == "all":
        selected = experiments
    else:
        selected = [exp for exp in experiments if exp["name"] == args.run]
        if not selected:
            raise SystemExit(f"Esperimento '{args.run}' non trovato.")

    for exp in selected:
        config = deep_merge(base_config, exp.get("overrides", {}))
        config["experiment"]["name"] = exp["name"]
        run_single_model(exp["model"], config, args.resume)


def run_single_model(model_name, config, resume):
    if model_name == "yolov11":
        yolo_model = build_yolo(config["models"]["yolov11"])
        train_yolo(config, yolo_model)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=optimizer_cfg["min_lr"]
    )

    start_epoch = 0
    if resume:
        checkpoint_path = config["training"]["checkpoint_path"]
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Ripresa addestramento da epoca {start_epoch + 1}")

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
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
