import torch


def evaluate_detection(model, val_loader, device, metrics, label_offset=0):
    metrics.reset()
    total_loss = 0.0

    model.train()
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if label_offset != 0:
                for t in targets:
                    t["labels"] = (t["labels"] + label_offset).clamp(min=0)

            loss_dict = model(images, targets)
            total_loss += sum(loss_dict.values()).item()

            model.eval()
            preds = model(images)
            model.train()
            metrics.update(preds, targets)

    val_loss = total_loss / max(len(val_loader), 1)
    metric_values = metrics.compute()
    return val_loss, metric_values
