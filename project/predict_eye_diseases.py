from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from disease_definitions import DISEASES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def load_available_models(model_dir: Path, device: torch.device) -> dict[str, nn.Module]:
    loaded: dict[str, nn.Module] = {}
    for disease_key in DISEASES:
        model_path = model_dir / f"best_{disease_key}_efficientnet_b0.pt"
        if model_path.exists():
            loaded[disease_key] = load_model(model_path, device)
    if not loaded:
        raise FileNotFoundError(f"No trained disease models found in {model_dir}")
    return loaded


def predict_image(
    models_by_disease: dict[str, nn.Module],
    image_path: Path,
    transform,
    device: torch.device,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    disease_results: list[dict[str, object]] = []
    with torch.no_grad():
        for disease_key, model in models_by_disease.items():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            probability = float(probs[1].cpu())
            disease = DISEASES[disease_key]
            disease_results.append(
                {
                    "disease_key": disease_key,
                    "disease_name": disease["display_ko"],
                    "probability": probability,
                    "percent": round(probability * 100, 2),
                    "description": disease["description"],
                    "prevention": disease["prevention"],
                }
            )

    disease_results.sort(key=lambda row: float(row["probability"]), reverse=True)
    top = disease_results[0]
    return {
        "image_path": str(image_path),
        "top_disease_key": top["disease_key"],
        "top_disease_name": top["disease_name"],
        "top_probability": top["probability"],
        "top_percent": top["percent"],
        "description": top["description"],
        "prevention": top["prevention"],
        "diseases": disease_results,
    }


def write_predictions(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "disease_key", "disease_name", "probability", "percent"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for disease in row["diseases"]:
                writer.writerow(
                    {
                        "image_path": row["image_path"],
                        "disease_key": disease["disease_key"],
                        "disease_name": disease["disease_name"],
                        "probability": disease["probability"],
                        "percent": disease["percent"],
                    }
                )


def print_prediction(row: dict[str, object]) -> None:
    print(f"image: {row['image_path']}")
    print(f"top: {row['top_disease_name']} ({float(row['top_probability']):.4f})")
    for disease in row["diseases"]:
        print(f"  {disease['disease_name']}: {float(disease['probability']):.4f}")
    print("-" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict multiple pet eye disease probabilities.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("project/outputs"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    models_by_disease = load_available_models(args.model_dir, device)
    transform = build_transform()
    images = collect_images(args.image)
    rows = [predict_image(models_by_disease, image_path, transform, device) for image_path in images]

    for row in rows:
        print_prediction(row)

    if args.output is not None:
        write_predictions(rows, args.output)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
