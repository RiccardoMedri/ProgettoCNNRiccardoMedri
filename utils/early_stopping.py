import numpy as np
import torch

from utils.alert import alert
from utils.beep import beep


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode="min", checkpoint_path="models/best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = np.Inf if mode == "min" else -np.Inf

    def __call__(self, metric_value, model, optimizer, epoch, scheduler=None):
        score = metric_value if self.mode == "max" else -metric_value

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model, optimizer, epoch, scheduler)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"Early stopping counter: {self.counter} / {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping attivato: Fine allenamento\n")
                alert()
        else:
            self.best_score = score
            self.save_checkpoint(metric_value, model, optimizer, epoch, scheduler)
            self.counter = 0

    def save_checkpoint(self, metric_value, model, optimizer, epoch, scheduler=None):
        print(
            f"Checkpoint salvato. Miglioramento metrica ({self.best_value:.4f} -> {metric_value:.4f}).\n"
        )
        beep()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_value": metric_value,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, self.checkpoint_path)
        self.best_value = metric_value
