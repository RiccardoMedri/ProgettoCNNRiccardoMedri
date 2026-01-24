import os
from typing import Dict

from PIL import Image


def convert_visdrone_to_yolo(images_dir, annotations_dir, labels_dir, class_map):
    os.makedirs(labels_dir, exist_ok=True)

    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        base_name = os.path.splitext(image_file)[0]
        ann_path = os.path.join(annotations_dir, f"{base_name}.txt")
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        if not os.path.exists(ann_path):
            open(label_path, "w").close()
            continue

        img_path = os.path.join(images_dir, image_file)
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                x, y, w, h, score, category, truncation, occlusion = parts[:8]
                category = int(category)
                if category <= 0:
                    continue

                if category not in class_map:
                    continue
                mapped = class_map[category] - 1
                if mapped < 0:
                    continue

                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)

                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                yolo_lines.append(f"{mapped} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_path, "w") as out:
            out.write("\n".join(yolo_lines))


def write_yolo_yaml(output_path: str, dataset_root: str, class_names: Dict[int, str]):
    names = [class_names[idx] for idx in sorted(class_names.keys())]
    content = (
        f"path: {dataset_root}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    with open(output_path, "w") as f:
        f.write(content)
