from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {
    0: "normal",
    1: "conjunctivitis_suspected",
}


def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]

    if input_path.is_dir():
        images = [
            p
            for p in sorted(input_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            raise ValueError(f"No images found under: {input_path}")
        return images

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = build_model()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(
    model: nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
) -> dict[str, str | float]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    normal_prob = float(probs[0].cpu())
    suspected_prob = float(probs[1].cpu())
    pred_label = int(torch.argmax(probs).cpu())

    return {
        "image_path": str(image_path),
        "pred_label": pred_label,
        "pred_name": CLASS_NAMES[pred_label],
        "normal_probability": normal_prob,
        "conjunctivitis_probability": suspected_prob,
    }


def write_predictions(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "pred_label",
        "pred_name",
        "normal_probability",
        "conjunctivitis_probability",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_prediction(row: dict[str, str | float]) -> None:
    print(f"image: {row['image_path']}")
    print(f"prediction: {row['pred_name']} ({int(row['pred_label'])})")
    print(f"normal_probability: {float(row['normal_probability']):.4f}")
    print(f"conjunctivitis_probability: {float(row['conjunctivitis_probability']):.4f}")
    print("-" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict conjunctivitis probability for pet eye images."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Image file or folder containing images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("project/outputs/best_conjunctivitis_efficientnet_b0.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for prediction results.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model, device)
    transform = build_transform()
    images = collect_images(args.image)

    rows = [predict_image(model, image_path, transform, device) for image_path in images]
    for row in rows:
        print_prediction(row)

    if args.output is not None:
        write_predictions(rows, args.output)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
