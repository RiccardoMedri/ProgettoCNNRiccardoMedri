import csv
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from codes.evaluate_detection import evaluate_detection
from codes.metrics import DetectionMetrics
from utils.class_names import class_names
from utils.early_stopping import EarlyStopping
from utils.time_manager import calculate_epoch_time, format_total_time, get_current_time
from utils.visualization import save_prediction_samples


def train_detection(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config,
    device,
    start_epoch=0,
    label_offset=0,
    model_name="model",
):
    runs_dir = config["training"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config["training"]["checkpoint_path"]), exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{config['experiment']['name']}"
    run_dir = os.path.join(runs_dir, model_name, run_name)
    tb_dir = os.path.join(run_dir, "tb")
    writer = SummaryWriter(tb_dir)
    results_csv = os.path.join(run_dir, "results.csv")
    epochs = config["training"]["epochs"]
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        delta=config["training"]["delta"],
        mode=config["training"]["early_stop_mode"],
        checkpoint_path=config["training"]["checkpoint_path"],
    )
    sum_time = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["use_amp"])
    metrics = DetectionMetrics(iou_threshold=config["evaluation"]["iou_threshold"])

    model.to(device)
    model.train()

    try:
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            running_loss = 0.0
            model.train()

            for step, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if label_offset != 0:
                    for t in targets:
                        t["labels"] = (t["labels"] + label_offset).clamp(min=0)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=config["training"]["use_amp"]):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

                scaler.scale(loss).backward()
                if config["training"]["gradient_clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["gradient_clip_norm"]
                    )
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                if step % config["training"]["log_interval"] == 0:
                    print(
                        f"[Epoch {epoch + 1}/{epochs}] Step {step} "
                        f"Loss: {loss.item():.4f}"
                    )

            train_loss = running_loss / max(len(train_loader), 1)
            val_loss, val_metrics = evaluate_detection(
                model, val_loader, device, metrics, label_offset=label_offset
            )

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            map_50 = float(val_metrics.get("map_50", 0.0))
            map_all = float(val_metrics.get("map", 0.0))
            precision = float(val_metrics.get("precision", 0.0))
            recall = float(val_metrics.get("recall", 0.0))
            f1 = 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)

            writer.add_scalar("Metrics/mAP50", map_50, epoch)
            writer.add_scalar("Metrics/mAP", map_all, epoch)
            writer.add_scalar("Metrics/Precision", precision, epoch)
            writer.add_scalar("Metrics/Recall", recall, epoch)
            writer.add_scalar("Metrics/F1", f1, epoch)

            map_per_class = val_metrics.get("map_per_class", None)
            if map_per_class is not None:
                for class_name, ap in zip(class_names, map_per_class.detach().cpu().tolist()):
                    writer.add_scalar(f"Metrics/AP/{class_name}", ap, epoch)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

            epoch_time = calculate_epoch_time(start_time)
            sum_time += epoch_time
            current_time = get_current_time()

            log_line = (
                f"Epoca [{epoch + 1}/{epochs}]:\n"
                f"- Train Loss: {train_loss:.4f}\n"
                f"- Val Loss: {val_loss:.4f}\n"
                f"- mAP@0.5: {map_50:.4f}\n"
                f"- mAP@0.5:0.95: {map_all:.4f}\n"
                f"- Precision: {precision:.4f}\n"
                f"- Recall: {recall:.4f}\n"
                f"- F1: {f1:.4f}\n"
                f"[ {current_time} - Tempo impiegato: {epoch_time:.2f}s ]"
            )
            print(log_line)

            write_results_row(
                results_csv,
                epoch + 1,
                train_loss,
                val_loss,
                map_50,
                map_all,
                precision,
                recall,
                f1,
                scheduler.get_last_lr()[0],
                epoch_time,
                map_per_class,
            )

            save_every = config["training"].get("save_predictions_every", 0)
            if save_every and (epoch + 1) % save_every == 0:
                pred_dir = os.path.join(run_dir, "predictions", f"epoch_{epoch + 1}")
                sample_count = config["training"].get("prediction_samples", 4)
                score_threshold = config["training"].get("prediction_score_threshold", 0.4)
                zero_based = label_offset == -1
                batch = next(iter(val_loader))
                save_prediction_samples(
                    model,
                    batch,
                    pred_dir,
                    config["data"]["normalize"],
                    class_names,
                    score_threshold=score_threshold,
                    max_samples=sample_count,
                    zero_based_labels=zero_based,
                    device=device,
                )

            scheduler.step(val_loss)
            stop_metric = val_loss
            if config["training"]["early_stop_metric"] == "map_50":
                stop_metric = float(val_metrics.get("map_50", 0.0))
            early_stopping(stop_metric, model, optimizer, epoch)
            if early_stopping.early_stop:
                break

    except KeyboardInterrupt:
        print("\nTraining interrotto manualmente.")

    finally:
        total_time_formatted = format_total_time(sum_time)
        print(f"\n[ Tempo totale impiegato: {total_time_formatted} ]")
        writer.flush()
        writer.close()


def write_results_row(
    csv_path,
    epoch,
    train_loss,
    val_loss,
    map_50,
    map_all,
    precision,
    recall,
    f1,
    lr,
    epoch_time,
    map_per_class,
):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            per_class_headers = [f"ap_{name}" for name in class_names]
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "map_50",
                    "map_50_95",
                    "precision",
                    "recall",
                    "f1",
                    "lr",
                    "epoch_time_sec",
                ]
                + per_class_headers
            )
        row = [epoch, train_loss, val_loss, map_50, map_all, precision, recall, f1, lr, epoch_time]
        if map_per_class is not None:
            per_class = map_per_class.detach().cpu().tolist()
        else:
            per_class = ["" for _ in class_names]
        writer.writerow(row + per_class)
