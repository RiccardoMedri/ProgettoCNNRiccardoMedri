from torch.utils.data import DataLoader

from data.detection_transforms import build_transforms
from data.visdrone_dataset import VisDroneDataset, collate_fn


def load_data(config):
    train_cfg = config["data"]["train"]
    val_cfg = config["data"]["val"]

    valid_categories = config["data"].get("valid_categories", None)
    train_dataset = VisDroneDataset(
        images_dir=train_cfg["images_dir"],
        annotations_dir=train_cfg["annotations_dir"],
        transforms=build_transforms(config, train=True),
        class_map=config["data"]["class_map"],
        valid_categories=valid_categories,
    )

    val_dataset = VisDroneDataset(
        images_dir=val_cfg["images_dir"],
        annotations_dir=val_cfg["annotations_dir"],
        transforms=build_transforms(config, train=False),
        class_map=config["data"]["class_map"],
        valid_categories=valid_categories,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
