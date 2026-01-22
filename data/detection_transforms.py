import random
from typing import Tuple

import torch
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        image = F.resize(image, self.size)
        new_w, new_h = image.size

        if "boxes" in target and target["boxes"].numel() > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
            target["boxes"] = boxes
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            w, _ = image.size
            if "boxes" in target and target["boxes"].numel() > 0:
                boxes = target["boxes"]
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
                target["boxes"] = boxes
        return image, target


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        image = F.adjust_brightness(image, 1 + (random.random() * 2 - 1) * self.brightness)
        image = F.adjust_contrast(image, 1 + (random.random() * 2 - 1) * self.contrast)
        image = F.adjust_saturation(image, 1 + (random.random() * 2 - 1) * self.saturation)
        image = F.adjust_hue(image, (random.random() * 2 - 1) * self.hue)
        return image, target


class RandomScale:
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        orig_w, orig_h = image.size
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        image = F.resize(image, (new_h, new_w))

        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / orig_h)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
            target["boxes"] = boxes
        return image, target


class RandomResize:
    def __init__(self, sizes):
        self.sizes = [int(s) for s in sizes]

    def __call__(self, image, target):
        size = random.choice(self.sizes)
        orig_w, orig_h = image.size
        image = F.resize(image, (size, size))
        new_w, new_h = image.size

        if "boxes" in target and target["boxes"].numel() > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
            target["boxes"] = boxes
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        return F.normalize(image, mean=self.mean, std=self.std), target


def build_transforms(config, train=True):
    size = config["data"]["image_size"]
    mean = config["data"]["normalize"]["mean"]
    std = config["data"]["normalize"]["std"]

    transforms = []
    if train:
        aug = config["data"]["augmentation"]
        if aug.get("random_scale", False):
            transforms.append(
                RandomScale(
                    min_scale=aug.get("scale_min", 0.8),
                    max_scale=aug.get("scale_max", 1.2),
                )
            )
        if aug.get("horizontal_flip", True):
            transforms.append(RandomHorizontalFlip(p=aug.get("flip_prob", 0.5)))
        if aug.get("color_jitter", True):
            transforms.append(
                ColorJitter(
                    brightness=aug.get("brightness", 0.2),
                    contrast=aug.get("contrast", 0.2),
                    saturation=aug.get("saturation", 0.2),
                    hue=aug.get("hue", 0.05),
                )
            )
        if aug.get("multi_scale", False):
            sizes = aug.get("multi_scale_sizes", [size])
            transforms.append(RandomResize(sizes))
        else:
            transforms.append(Resize((size, size)))
    else:
        transforms.append(Resize((size, size)))
    transforms.extend([ToTensor(), Normalize(mean, std)])
    return Compose(transforms)
