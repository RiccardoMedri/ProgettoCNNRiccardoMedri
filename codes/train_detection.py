import csv
import json
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


#Loop di training completo per modelli di detection con logging e early stopping
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
    run_dir=None,
):
    runs_dir = config["training"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    #Prepara cartelle di run e TensorBoard
    if run_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{config['experiment']['name']}"
        run_dir = os.path.join(runs_dir, "model", run_name)
    tb_dir = os.path.join(run_dir, "tb")
    writer = SummaryWriter(tb_dir)
    results_csv = os.path.join(run_dir, "results.csv")
    checkpoint_path = os.path.join(run_dir, "best_model.pth")
    config_path = os.path.join(run_dir, "run_config.json")

    #Salva una copia della config usata per la run
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    epochs = config["training"]["epochs"]

    #Early stopping basato sulla metrica scelta
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        delta=config["training"]["delta"],
        mode=config["training"]["early_stop_mode"],
        checkpoint_path=checkpoint_path,
    )
    sum_time = 0.0

    #MP per velocizzare training su GPU
    scaler_device = "cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=config["training"]["use_amp"])

    #Metriche di detection con soglie da config
    metrics = DetectionMetrics(
        iou_threshold=config["evaluation"]["iou_threshold"],
        score_threshold=config["evaluation"]["score_threshold"],
    )
    #Accumulazione dei gradienti per simulare batch piu grandi
    accum_steps = max(int(config["training"].get("gradient_accumulation_steps", 1)), 1)

    #Sposta modello su device
    model.to(device)
    model.train()

    try:
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            running_loss = 0.0
            model.train()
            optimizer.zero_grad(set_to_none=True)

            for step, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if label_offset != 0:
                    for t in targets:
                        t["labels"] = (t["labels"] + label_offset).clamp(min=0)

                #Forward con AMP e accumulo gradiente
                with torch.amp.autocast("cuda", enabled=config["training"]["use_amp"]):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                    loss_value = loss.item()
                    loss = loss / accum_steps

                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    #Clip dei gradienti per stabilità
                    if config["training"]["gradient_clip_norm"] is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config["training"]["gradient_clip_norm"]
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss_value

                #Log periodico dello step
                if step % config["training"]["log_interval"] == 0:
                    print(
                        f"[Epoch {epoch + 1}/{epochs}] Step {step} "
                        f"Loss: {loss_value:.4f}"
                    )

            #Valutazione su validation set
            train_loss = running_loss / max(len(train_loader), 1)
            val_loss, val_metrics, _ = evaluate_detection(
                model,
                val_loader,
                device,
                metrics,
                label_offset=label_offset,
                track_inference_time=config["evaluation"].get("track_inference_time", False),
                compute_loss=True,
            )

            #Log scalari su TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            if val_loss is not None:
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
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

            epoch_time = calculate_epoch_time(start_time)
            sum_time += epoch_time
            current_time = get_current_time()

            #Log testuale di riepilogo epoca
            log_parts = [
                f"Epoca [{epoch + 1}/{epochs}]:",
                f"- Train Loss: {train_loss:.4f}",
            ]
            if val_loss is not None:
                log_parts.append(f"- Val Loss: {val_loss:.4f}")
            log_parts.extend(
                [
                    f"- mAP@0.5: {map_50:.4f}",
                    f"- mAP@0.5:0.95: {map_all:.4f}",
                    f"- Precision: {precision:.4f}",
                    f"- Recall: {recall:.4f}",
                    f"- F1: {f1:.4f}",
                    f"[ {current_time} - Tempo impiegato: {epoch_time:.2f}s ]",
                ]
            )
            log_line = "\n".join(log_parts)
            print(log_line)

            #Salva riga risultati su CSV
            write_results_row(
                results_csv,
                epoch + 1,
                train_loss,
                val_loss if val_loss is not None else "",
                map_50,
                map_all,
                precision,
                recall,
                f1,
                scheduler.get_last_lr()[0],
                epoch_time,
            )

            #Salva campioni di predizione a intervalli
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
                    apply_unnorm=False,
                )

            #Step scheduler e controllo early stopping
            scheduler.step()
            stop_metric = None
            if config["training"]["early_stop_metric"] == "map_50":
                stop_metric = float(val_metrics.get("map_50", 0.0))
            elif config["training"]["early_stop_metric"] == "map_50_95":
                stop_metric = float(val_metrics.get("map", 0.0))
            elif val_loss is not None:
                stop_metric = val_loss
            else:
                raise SystemExit(
                    "Val Loss non disponibile: imposta training.early_stop_metric su map_50 o map_50_95."
                )
            early_stopping(stop_metric, model, optimizer, epoch, scheduler)
            if early_stopping.early_stop:
                break

    except KeyboardInterrupt:
        print("\nTraining interrotto manualmente.")

    finally:
        #Chiusura e log tempo totale
        total_time_formatted = format_total_time(sum_time)
        print(f"\n[ Tempo totale impiegato: {total_time_formatted} ]")
        writer.flush()
        writer.close()


#Scrive una riga nel CSV risultati, creando header se serve
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
):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
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
            )
        row = [
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
        ]
        writer.writerow(row)
