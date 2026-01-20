import argparse
import json
import os

import torch
from PIL import Image, ImageDraw

from data.detection_transforms import build_transforms
from models.detectors import build_detector
from utils.checkpoints import load_checkpoint
from utils.class_names import class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Inference su immagini VisDrone")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--model", required=True, choices=["retinanet", "faster_rcnn"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def draw_boxes(image, outputs, score_threshold, zero_based_labels=False):
    draw = ImageDraw.Draw(image)
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if zero_based_labels:
            class_label = class_names[label] if 0 <= label < len(class_names) else f"id-{label}"
        else:
            class_label = class_names[label - 1] if 0 < label <= len(class_names) else f"id-{label}"
        text = f"{class_label} {score:.2f}"
        draw.text((x1, y1), text, fill="red")

    return image


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    model_cfg = config["models"][args.model]
    model = build_detector(model_cfg, num_classes=config["data"]["num_classes"])
    load_checkpoint(model, None, config["training"]["checkpoint_path"])
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    transforms = build_transforms(config, train=False)
    dummy_target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }
    resized_image, _ = transforms(image, dummy_target)
    resized_pil = tensor_to_pil(resized_image, config["data"]["normalize"])

    with torch.no_grad():
        outputs = model([resized_image.to(device)])[0]

    zero_based = args.model == "retinanet"
    output_image = draw_boxes(resized_pil, outputs, args.score_threshold, zero_based_labels=zero_based)
    output_path = os.path.splitext(args.image)[0] + "_det.png"
    output_image.save(output_path)
    print(f"Output salvato in {output_path}")


def tensor_to_pil(tensor_image, normalize_cfg):
    mean = torch.tensor(normalize_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(normalize_cfg["std"]).view(3, 1, 1)
    image = tensor_image.cpu() * std + mean
    image = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)


if __name__ == "__main__":
    main()
