from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
from pathlib import Path, PureWindowsPath

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from prepare_disease_type_csv import CLASS_TO_ID


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, split: str, project_root: Path, transform=None):
        self.transform = transform
        self.project_root = project_root
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [row for row in rows if row["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.resolve_image_path(row)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(int(row["class_id"]), dtype=torch.long)

    def resolve_image_path(self, row: dict[str, str]) -> Path:
        image_path = Path(row["image_path"])
        if image_path.exists():
            return image_path
        relative_parts = PureWindowsPath(row["relative_path"]).parts
        image_path = self.project_root.joinpath(*relative_parts)
        if image_path.exists():
            return image_path
        raise FileNotFoundError(f"Missing image: {row['relative_path']}")


def build_model(num_classes: int, freeze_features: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(512, num_classes),
    )
    return model


def make_sampler(dataset: CsvImageDataset) -> WeightedRandomSampler:
    labels = [int(row["class_id"]) for row in dataset.rows]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for true, pred in zip(labels.cpu(), preds.cpu()):
                confusion[int(true), int(pred)] += 1

    f1_scores = []
    for idx in range(num_classes):
        tp = confusion[idx, idx].item()
        fp = confusion[:, idx].sum().item() - tp
        fn = confusion[idx, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "class_f1": f1_scores,
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_classes = len(CLASS_TO_ID)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CsvImageDataset(args.csv, "train", args.project_root, train_tfms)
    val_ds = CsvImageDataset(args.csv, "val", args.project_root, eval_tfms)
    test_ds = CsvImageDataset(args.csv, "test", args.project_root, eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_ds),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes, args.freeze_features).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_path = args.output_dir / "best_eye_disease_type_efficientnet_b0.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_metrics = evaluate(model, val_loader, device, num_classes)
        print(
            f"epoch {epoch:02d} train_loss={running_loss / seen:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)

    model.load_state_dict(best_state)
    print("test metrics:", evaluate(model, test_loader, device, num_classes), flush=True)
    print("saved:", best_path, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("project/datasets/eye_disease_type_dataset.csv"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("project/outputs"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-features", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
