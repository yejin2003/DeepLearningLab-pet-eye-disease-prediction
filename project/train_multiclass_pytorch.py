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

from prepare_multiclass_csv import CLASS_TO_ID


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
        label = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return image, label

    def resolve_image_path(self, row: dict[str, str]) -> Path:
        image_path = Path(row["image_path"])
        if image_path.exists():
            return image_path

        relative_parts = PureWindowsPath(row["relative_path"]).parts
        image_path = self.project_root.joinpath(*relative_parts)
        if image_path.exists():
            return image_path

        raise FileNotFoundError(f"Missing image: {row['relative_path']}")


def build_model(num_classes: int) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(2048, num_classes),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_TO_ID)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.85, 1.2)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
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
    sampler = make_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.999,
        nesterov=True,
        weight_decay=1e-5,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_path = args.output_dir / "best_eye_multiclass_efficientnet_b0.pt"

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
            f"epoch {epoch:02d} "
            f"train_loss={running_loss / seen:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, num_classes)
    print("test metrics:", test_metrics)
    print("saved:", best_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("project/datasets/eye_multiclass_dataset.csv"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("project/outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
