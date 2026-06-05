from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MIN_INPUT_SIZE = 224


def _check(label: str, ok: bool, message: str) -> dict[str, object]:
    return {
        "label": label,
        "ok": ok,
        "status": "통과" if ok else "주의",
        "message": message,
    }


def check_image_quality(image_path: Path) -> dict[str, object]:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    contrast = float(gray.std())
    dark_ratio = float((gray < 70).mean())
    bright_ratio = float((gray > 245).mean())

    resolution_ok = min(width, height) >= MIN_INPUT_SIZE
    eye_region_ok = 0.008 <= dark_ratio <= 0.70 and contrast >= 18.0
    contrast_ok = contrast >= 22.0
    exposure_ok = bright_ratio < 0.70

    checks = [
        _check(
            "해상도",
            resolution_ok,
            f"원본 크기 {width} x {height}px, 모델 입력 기준 {MIN_INPUT_SIZE}px 이상 확인",
        ),
        _check(
            "눈 영역",
            eye_region_ok,
            "동공/각막으로 보이는 어두운 영역과 주변 명암 차이를 확인",
        ),
        _check(
            "명암",
            contrast_ok,
            "눈 주변 특징을 구분할 수 있는 밝기 차이를 확인",
        ),
        _check(
            "노출",
            exposure_ok,
            "과도하게 밝아 세부 특징이 사라지는지 확인",
        ),
    ]
    passed = sum(1 for item in checks if item["ok"])

    return {
        "width": width,
        "height": height,
        "contrast": round(contrast, 2),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "passed": passed,
        "total": len(checks),
        "quality_ok": passed >= 3,
        "summary": "분석 적합" if passed >= 3 else "촬영 품질 주의",
        "checks": checks,
    }
