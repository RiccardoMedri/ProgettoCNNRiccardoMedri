import os

import torch
from PIL import Image, ImageDraw


def tensor_to_pil(tensor_image, normalize_cfg=None, apply_unnorm=True):
    image = tensor_image.cpu()
    if apply_unnorm and normalize_cfg is not None:
        mean = torch.tensor(normalize_cfg["mean"]).view(3, 1, 1)
        std = torch.tensor(normalize_cfg["std"]).view(3, 1, 1)
        image = image * std + mean
    image = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def draw_detections(image, outputs, class_names, score_threshold, zero_based_labels=False, color="red"):
    draw = ImageDraw.Draw(image)
    boxes = outputs.get("boxes", torch.zeros((0, 4))).cpu().numpy()
    scores = outputs.get("scores", torch.zeros((0,))).cpu().numpy()
    labels = outputs.get("labels", torch.zeros((0,))).cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if zero_based_labels:
            class_label = class_names[label] if 0 <= label < len(class_names) else f"id-{label}"
        else:
            class_label = class_names[label - 1] if 0 < label <= len(class_names) else f"id-{label}"
        draw.text((x1, y1), f"{class_label} {score:.2f}", fill=color)
    return image


def draw_targets(image, targets, class_names, zero_based_labels=False, color="green"):
    draw = ImageDraw.Draw(image)
    boxes = targets.get("boxes", torch.zeros((0, 4))).cpu().numpy()
    labels = targets.get("labels", torch.zeros((0,))).cpu().numpy()

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if zero_based_labels:
            class_label = class_names[label] if 0 <= label < len(class_names) else f"id-{label}"
        else:
            class_label = class_names[label - 1] if 0 < label <= len(class_names) else f"id-{label}"
        draw.text((x1, y1), class_label, fill=color)
    return image


def save_prediction_samples(
    model,
    batch,
    output_dir,
    normalize_cfg,
    class_names,
    score_threshold=0.4,
    max_samples=4,
    zero_based_labels=False,
    device="cpu",
    apply_unnorm=True,
):
    os.makedirs(output_dir, exist_ok=True)
    images, targets = batch
    images = images[:max_samples]
    targets = targets[:max_samples]

    images_device = [img.to(device) for img in images]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(images_device)
    if was_training:
        model.train()

    for idx, (img_tensor, output, target) in enumerate(zip(images, outputs, targets)):
        image = tensor_to_pil(img_tensor, normalize_cfg, apply_unnorm=apply_unnorm)
        image = draw_targets(image, target, class_names, zero_based_labels, color="green")
        image = draw_detections(image, output, class_names, score_threshold, zero_based_labels, color="red")
        image.save(os.path.join(output_dir, f"sample_{idx + 1}.png"))
