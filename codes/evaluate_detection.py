import time
import torch
from torch import nn

#Nei modelli di detection di Torchvision la forward funziona:
# - model.eval() + model(images) → ritorna solo prediction
# - model.train() + model(images, targets) → ritorna un dizionare di loss
#Se tenessi il modello solo in eval(), non potrei calcolare la loss di validazione,
#perché il modello non accetterebbe e non produrrebbe le losses


#Scansiona tutti i moduli del modello ed individua quelli di tipo BatchNorm e Dropout
#Salvo lo stato in una lista di tuple e mette il modulo in eval
def _freeze_bn_dropout(model):
    frozen = []
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            frozen.append((module, module.training))
            module.eval()
    return frozen


#Ripristino lo stato train/eval (in base al flag) dei moduli congelati nella funzione _freeze_bn_dropout
#Valuto il modello su val_loader con metriche, loss opzionale e calcolo del tempo di inferenza
def evaluate_detection(
    model,
    val_loader,
    device,
    metrics,
    label_offset=0,
    track_inference_time=True,
    compute_loss=False,
):
    metrics.reset()
    total_pred_time = 0.0
    total_images = 0
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if label_offset != 0:
                for t in targets:
                    t["labels"] = (t["labels"] + label_offset).clamp(min=0)

            if compute_loss:
                model.train()
                frozen = _freeze_bn_dropout(model)
                loss_dict = model(images, targets)
                total_loss += sum(loss_dict.values()).item()
                model.eval()

            if track_inference_time:
                start = time.perf_counter()
                preds = model(images)
                total_pred_time += time.perf_counter() - start
                total_images += len(images)
            else:
                preds = model(images)

            metrics.update(preds, targets)

    metric_values = metrics.compute()
    avg_pred_time = total_pred_time / max(total_images, 1) if track_inference_time else None
    val_loss = total_loss / max(len(val_loader), 1) if compute_loss else None
    return val_loss, metric_values, avg_pred_time
