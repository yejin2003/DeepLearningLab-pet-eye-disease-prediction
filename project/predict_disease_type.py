from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from disease_definitions import DISEASES
from prepare_disease_type_csv import CLASS_TO_ID, ID_TO_CLASS


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(512, num_classes),
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
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()

    diseases = []
    for idx, probability in enumerate(probs):
        disease_key = ID_TO_CLASS[idx]
        disease = DISEASES[disease_key]
        prob = float(probability)
        diseases.append(
            {
                "disease_key": disease_key,
                "disease_name": disease["display_ko"],
                "probability": prob,
                "percent": round(prob * 100, 2),
                "description": disease["description"],
                "prevention": disease["prevention"],
            }
        )
    diseases.sort(key=lambda row: float(row["probability"]), reverse=True)
    return {
        "image_path": str(image_path),
        "top_disease_key": diseases[0]["disease_key"],
        "top_disease_name": diseases[0]["disease_name"],
        "top_probability": diseases[0]["probability"],
        "top_percent": diseases[0]["percent"],
        "description": diseases[0]["description"],
        "prevention": diseases[0]["prevention"],
        "diseases": diseases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("project/outputs/best_eye_disease_type_efficientnet_b0.pt"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model, device)
    transform = build_transform()
    result = predict_image(model, args.image, transform, device)
    print(f"image: {result['image_path']}")
    print(f"top: {result['top_disease_name']} ({result['top_percent']}%)")
    for disease in result["diseases"]:
        print(f"  {disease['disease_name']}: {disease['percent']}%")


if __name__ == "__main__":
    main()
