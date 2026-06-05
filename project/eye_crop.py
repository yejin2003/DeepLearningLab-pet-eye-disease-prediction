from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def crop_eye_region(image_path: Path, output_path: Path | None = None, margin: float = 0.55) -> Path:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    gray = arr.mean(axis=2)

    # The eye/cornea is usually one of the darkest connected-looking regions in
    # close pet eye photos. This simple detector keeps the project dependency
    # light while making external phone photos closer to the AIHub eye crops.
    threshold = np.percentile(gray, 18)
    mask = gray <= threshold

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image_path

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    pad_x = int(width * margin)
    pad_y = int(height * margin)

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(image.width, x2 + pad_x)
    bottom = min(image.height, y2 + pad_y)

    # Avoid tiny or extreme crops if the threshold selected noise.
    crop_width = right - left
    crop_height = bottom - top
    if crop_width < image.width * 0.25 or crop_height < image.height * 0.25:
        return image_path

    cropped = image.crop((left, top, right, bottom))
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_eye_crop{image_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path)
    return output_path
