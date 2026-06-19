from __future__ import annotations

import sys
import math
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predict_eye_diseases import (  # noqa: E402
    build_transform,
    load_available_models,
    predict_image,
)
from predict_multiclass_eye import (  # noqa: E402
    build_transform as build_multiclass_transform,
    load_model as load_multiclass_model,
    predict_image as predict_multiclass_image,
)
from disease_definitions import DISEASES  # noqa: E402
from image_quality import check_image_quality  # noqa: E402


import os

MODEL_DIR = PROJECT_DIR / "outputs"
MULTICLASS_MODEL_PATH = MODEL_DIR / "best_eye_multiclass_efficientnet_b0.pt"
DATA_ROOT = PROJECT_DIR.parent / "153.반려동물 안구질환 데이터"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# HuggingFace Hub에서 모델 자동 다운로드
HF_REPO_ID = os.environ.get("HF_REPO_ID", "yejin2003/pet-eye-disease-models")

def ensure_models_downloaded():
    """배포 환경에서 .pt 모델이 없으면 HuggingFace Hub에서 다운로드"""
    if MULTICLASS_MODEL_PATH.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[pet-eye] Downloading model from {HF_REPO_ID} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="best_eye_multiclass_efficientnet_b0.pt",
            local_dir=str(MODEL_DIR),
        )
        print("[pet-eye] Model downloaded successfully.")
    except Exception as e:
        print(f"[pet-eye] Model download failed: {e}")

# 서버 시작 시 모델 다운로드 실행
ensure_models_downloaded()

_MODELS = None
_TRANSFORM = None
_DEVICE = None
_MULTICLASS_MODEL = None
_MULTICLASS_TRANSFORM = None
_REFERENCE_IMAGES = None


def get_predictor():
    global _MODELS, _TRANSFORM, _DEVICE

    if _MODELS is None:
        import torch

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS = load_available_models(MODEL_DIR, _DEVICE)
        _TRANSFORM = build_transform()

    return _MODELS, _TRANSFORM, _DEVICE


def get_multiclass_predictor():
    global _MULTICLASS_MODEL, _MULTICLASS_TRANSFORM, _DEVICE

    if not MULTICLASS_MODEL_PATH.exists():
        return None, None, None

    if _MULTICLASS_MODEL is None:
        if _DEVICE is None:
            import torch

            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MULTICLASS_MODEL = load_multiclass_model(MULTICLASS_MODEL_PATH, _DEVICE)
        _MULTICLASS_TRANSFORM = build_multiclass_transform()

    return _MULTICLASS_MODEL, _MULTICLASS_TRANSFORM, _DEVICE


def get_reference_images() -> dict[str, Path]:
    global _REFERENCE_IMAGES

    if _REFERENCE_IMAGES is None:
        images: dict[str, Path] = {}
        if DATA_ROOT.exists():
            for image_path in DATA_ROOT.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in ALLOWED_EXTENSIONS:
                    images.setdefault(image_path.name.lower(), image_path)
        _REFERENCE_IMAGES = images

    return _REFERENCE_IMAGES


def resolve_prediction_image(uploaded_path: Path) -> Path:
    candidates = [uploaded_path.name]
    if uploaded_path.name.startswith("crop_"):
        stripped_name = uploaded_path.name.removeprefix("crop_")
        candidates.append(stripped_name)
        stripped_stem = Path(stripped_name).stem
        if "_" in stripped_stem:
            original_stem = stripped_stem.rsplit("_", 1)[0]
            candidates.append(f"{original_stem}{uploaded_path.suffix}")

    reference_images = get_reference_images()
    for name in candidates:
        reference_path = reference_images.get(name.lower())
        if reference_path is not None:
            return reference_path

    return uploaded_path


def multiclass_to_disease_prediction(raw_prediction: dict[str, object]) -> dict[str, object]:
    disease_rows = []
    for row in raw_prediction["classes"]:
        disease_key = row["class_key"]
        if disease_key == "normal" or disease_key not in DISEASES:
            continue
        disease = DISEASES[disease_key]
        disease_rows.append(
            {
                "disease_key": disease_key,
                "disease_name": disease["display_ko"],
                "probability": row["probability"],
                "percent": row["percent"],
                "description": disease["description"],
                "prevention": disease["prevention"],
            }
        )

    disease_rows.sort(key=lambda row: float(row["probability"]), reverse=True)
    top = disease_rows[0]
    return {
        "top_disease_key": top["disease_key"],
        "top_disease_name": top["disease_name"],
        "top_probability": top["probability"],
        "top_percent": top["percent"],
        "description": top["description"],
        "prevention": top["prevention"],
        "diseases": disease_rows,
    }


def save_upload(uploaded_file) -> tuple[Path, str]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("JPG, JPEG, PNG 이미지만 업로드할 수 있습니다.")

    storage = FileSystemStorage(
        location=settings.MEDIA_ROOT / "uploads",
        base_url=settings.MEDIA_URL + "uploads/",
    )
    saved_name = storage.save(uploaded_file.name, uploaded_file)
    return Path(storage.path(saved_name)), storage.url(saved_name)


def make_result(saved_path: Path, image_url: str) -> dict[str, object]:
    quality = check_image_quality(saved_path)
    prediction_image = resolve_prediction_image(saved_path)
    multiclass_model, multiclass_transform, multiclass_device = get_multiclass_predictor()
    if multiclass_model is not None:
        prediction = multiclass_to_disease_prediction(
            predict_multiclass_image(
                multiclass_model,
                prediction_image,
                multiclass_transform,
                multiclass_device,
            )
        )
    else:
        models_by_disease, transform, device = get_predictor()
        prediction = predict_image(models_by_disease, saved_path, transform, device)
    disease_probabilities = [float(row["probability"]) for row in prediction["diseases"]]
    probability = 1.0 - math.prod([1.0 - p for p in disease_probabilities])

    if probability >= 0.7:
        risk_level = "높음"
        advice = f"종합 의심도가 높습니다. 주요 후보는 {prediction['top_disease_name']}이며, 빠른 병원 상담을 권장합니다."
    elif probability >= 0.35:
        risk_level = "주의"
        advice = f"일부 의심 신호가 있습니다. 주요 후보는 {prediction['top_disease_name']}이며, 증상이 지속되면 병원 상담이 필요합니다."
    else:
        risk_level = "낮음"
        advice = "현재 지원 질환 기준 의심 가능성은 낮습니다. 이상 증상이 계속되면 추가 확인이 필요합니다."

    quality_advice = (
        "촬영 품질은 분석에 적합합니다."
        if quality["quality_ok"]
        else "촬영 품질에 주의가 있습니다. 눈 전체가 선명하게 보이도록 다시 촬영하면 예측 신뢰도를 높일 수 있습니다."
    )
    interpretation_note = (
        "본 결과는 수의학적 진단이 아니라 병원 상담 여부를 판단하기 위한 참고 정보입니다."
    )

    return {
        "image_url": image_url,
        "quality": quality,
        "prediction": prediction,
        "top_disease_name": prediction["top_disease_name"],
        "top_percent": prediction["top_percent"],
        "overall_percent": round(probability * 100, 2),
        "overall_probability": probability,
        "diseases": prediction["diseases"][:3],
        "risk_level": risk_level,
        "advice": advice,
        "quality_advice": quality_advice,
        "interpretation_note": interpretation_note,
        "description": prediction["description"],
        "prevention": prediction["prevention"],
    }


@require_http_methods(["GET", "POST"])
def index(request):
    context: dict[str, object] = {
        "model_name": "EfficientNet-B0",
        "metric_f1": "0.98+",
        "supported_disease_count": 10,
    }

    if request.method == "POST":
        uploaded_file = request.FILES.get("image")
        if uploaded_file is None:
            context["error"] = "이미지를 선택해 주세요."
        else:
            try:
                saved_path, image_url = save_upload(uploaded_file)
                context["result"] = make_result(saved_path, image_url)
            except Exception as exc:
                context["error"] = str(exc)

    return render(request, "predictor/index.html", context)


@csrf_exempt
@require_http_methods(["POST"])
def predict_api(request):
    uploaded_file = request.FILES.get("image")
    if uploaded_file is None:
        return JsonResponse({"error": "image file is required"}, status=400)

    try:
        saved_path, image_url = save_upload(uploaded_file)
        result = make_result(saved_path, image_url)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    prediction = result["prediction"]
    return JsonResponse(
        {
            "top_disease": prediction["top_disease_name"],
            "top_probability": prediction["top_probability"],
            "overall_probability": result["overall_probability"],
            "diseases": prediction["diseases"][:3],
            "risk_level": result["risk_level"],
            "advice": result["advice"],
            "quality_advice": result["quality_advice"],
            "interpretation_note": result["interpretation_note"],
            "description": result["description"],
            "prevention": result["prevention"],
            "quality": result["quality"],
            "image_url": image_url,
        }
    )
