import torch
from torchvision.transforms import v2 as T

#Definisce pipeline di trasformazioni e augmentation per detection
def build_transforms(config, train=True, use_model_internal_preprocessing=False):
    size = config["data"]["image_size"]
    mean = config["data"]["normalize"]["mean"]
    std = config["data"]["normalize"]["std"]

    transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if train:
        aug = config["data"]["augmentation"]
        if aug.get("horizontal_flip", True):
            transforms.append(T.RandomHorizontalFlip(p=aug.get("flip_prob", 0.5)))
        if aug.get("color_jitter", True):
            transforms.append(
                T.ColorJitter(
                    brightness=aug.get("brightness", 0.2),
                    contrast=aug.get("contrast", 0.2),
                    saturation=aug.get("saturation", 0.2),
                    hue=aug.get("hue", 0.05),
                )
            )
        #Effettua la resize solo se non si utilizza il preprocessing interno del modello    
        if not use_model_internal_preprocessing:
            transforms.append(T.Resize((size, size), antialias=True))
    else:
        if not use_model_internal_preprocessing:
            transforms.append(T.Resize((size, size), antialias=True))
    transforms.append(T.SanitizeBoundingBoxes())
    if not use_model_internal_preprocessing:
        transforms.append(T.Normalize(mean=mean, std=std))
    return T.Compose(transforms)
