from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from prepare_multiclass_csv import CLASS_TO_ID, class_display_name


ID_TO_CLASS = {idx: key for key, idx in CLASS_TO_ID.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(2048, num_classes),
    )
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
        return [input_path]
    return [
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = build_model(len(CLASS_TO_ID))
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(model: nn.Module, image_path: Path, transform, device: torch.device) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    classes = []
    for idx, probability in enumerate(probs):
        class_key = ID_TO_CLASS[idx]
        classes.append(
            {
                "class_id": idx,
                "class_key": class_key,
                "class_name": class_display_name(class_key),
                "probability": float(probability),
                "percent": round(float(probability) * 100, 2),
            }
        )
    classes.sort(key=lambda row: row["probability"], reverse=True)
    return {
        "image_path": str(image_path),
        "top": classes[0],
        "classes": classes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("project/outputs/best_eye_multiclass_efficientnet_b0.pt"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model, device)
    transform = build_transform()
    for image_path in collect_images(args.image):
        result = predict_image(model, image_path, transform, device)
        print(f"image: {result['image_path']}")
        print(f"top: {result['top']['class_name']} ({result['top']['percent']}%)")
        for row in result["classes"]:
            print(f"  {row['class_name']}: {row['percent']}%")
        print("-" * 60)


if __name__ == "__main__":
    main()
