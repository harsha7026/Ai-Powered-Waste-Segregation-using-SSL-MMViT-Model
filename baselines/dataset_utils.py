"""
Shared dataset utilities for baseline models.
Ensures all baselines use the SAME fixed train/val/test split files.
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Configuration (matching main project)
CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
IMAGE_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = False):
    """
    Get image preprocessing transforms (consistent with main project).
    
    Args:
        train: If True, returns training transforms with augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])


class SplitImageDataset(Dataset):
    """Dataset that reads samples from fixed split records."""

    def __init__(self, records, data_dir: Path, transform=None):
        self.records = records
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = self.data_dir / record["relative_path"]
        label = int(record["label"])

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def resolve_default_data_dir() -> Path:
    """Resolve default backend data/train directory."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "backend" / "data" / "train",
        script_dir.parent.parent / "waste-segmentation-project" / "backend" / "data" / "train",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve backend/data/train directory.")


def _build_records(data_dir: Path):
    image_folder = datasets.ImageFolder(root=str(data_dir))
    records = []
    for path, label in image_folder.samples:
        rel_path = str(Path(path).resolve().relative_to(data_dir.resolve())).replace("\\", "/")
        records.append({
            "relative_path": rel_path,
            "label": int(label),
            "class_name": image_folder.classes[label],
        })
    return records


def create_or_load_fixed_splits(
    data_dir: str,
    split_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Create (once) or load fixed stratified train/val/test split files.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    data_dir_path = Path(data_dir)
    split_dir_path = Path(split_dir)
    split_dir_path.mkdir(parents=True, exist_ok=True)

    train_file = split_dir_path / "train_split.json"
    val_file = split_dir_path / "val_split.json"
    test_file = split_dir_path / "test_split.json"
    meta_file = split_dir_path / "split_meta.json"

    if train_file.exists() and val_file.exists() and test_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            train_records = json.load(f)
        with open(val_file, "r", encoding="utf-8") as f:
            val_records = json.load(f)
        with open(test_file, "r", encoding="utf-8") as f:
            test_records = json.load(f)
        return train_records, val_records, test_records

    records = _build_records(data_dir_path)

    by_class = {name: [] for name in CLASS_NAMES}
    for rec in records:
        by_class[rec["class_name"]].append(rec)

    rng = random.Random(seed)
    train_records = []
    val_records = []
    test_records = []

    for class_name in CLASS_NAMES:
        class_records = by_class[class_name]
        rng.shuffle(class_records)

        n = len(class_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_records.extend(class_records[:n_train])
        val_records.extend(class_records[n_train:n_train + n_val])
        test_records.extend(class_records[n_train + n_val:n_train + n_val + n_test])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    meta = {
        "data_dir": str(data_dir_path),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return train_records, val_records, test_records


def get_dataloaders(
    data_dir: str,
    split_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Build train/val/test dataloaders from fixed split files.
    """
    train_records, val_records, test_records = create_or_load_fixed_splits(
        data_dir=data_dir,
        split_dir=split_dir,
    )

    data_dir_path = Path(data_dir)
    train_dataset = SplitImageDataset(train_records, data_dir_path, transform=get_transforms(train=True))
    val_dataset = SplitImageDataset(val_records, data_dir_path, transform=get_transforms(train=False))
    test_dataset = SplitImageDataset(test_records, data_dir_path, transform=get_transforms(train=False))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    print(f"Loaded data from: {data_dir}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {CLASS_NAMES}")

    return train_loader, val_loader, test_loader, dataset_sizes


def get_split_records(data_dir: str, split_dir: str):
    """Return fixed split records for train/val/test."""
    return create_or_load_fixed_splits(data_dir=data_dir, split_dir=split_dir)
