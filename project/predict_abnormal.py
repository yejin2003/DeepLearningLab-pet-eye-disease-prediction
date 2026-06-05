from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(2048, 2),
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
    model = build_model()
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
    return {
        "image_path": str(image_path),
        "normal_probability": float(probs[0]),
        "abnormal_probability": float(probs[1]),
        "abnormal_percent": round(float(probs[1]) * 100, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("project/outputs/best_dog_general_abnormal_efficientnet_b0.pt"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model, device)
    result = predict_image(model, args.image, build_transform(), device)
    print(result)


if __name__ == "__main__":
    main()
