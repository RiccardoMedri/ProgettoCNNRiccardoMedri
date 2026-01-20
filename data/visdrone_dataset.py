import os
from typing import Dict, List, Tuple

import torch
from PIL import Image


class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        transforms=None,
        class_map=None,
        valid_categories=None,
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.class_map = class_map or {}
        self.valid_categories = set(valid_categories) if valid_categories else None
        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

    def __len__(self):
        return len(self.image_files)

    def _parse_annotation(self, ann_path: str) -> Tuple[List[List[float]], List[int]]:
        boxes = []
        labels = []

        if not os.path.exists(ann_path):
            return boxes, labels

        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                x, y, w, h, score, category, truncation, occlusion = parts[:8]
                category = int(category)

                if category <= 0:
                    continue
                if self.valid_categories and category not in self.valid_categories:
                    continue

                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(self.class_map.get(category, category))

        return boxes, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        ann_path = os.path.join(self.annotations_dir, os.path.splitext(image_name)[0] + ".txt")

        image = Image.open(image_path).convert("RGB")
        boxes, labels = self._parse_annotation(ann_path)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        if boxes_tensor.numel() > 0:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((labels_tensor.shape[0],), dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
