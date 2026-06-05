from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from disease_definitions import positive_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT
    / "153.\ubc18\ub824\ub3d9\ubb3c \uc548\uad6c\uc9c8\ud658 \ub370\uc774\ud130"
    / "01.\ub370\uc774\ud130"
    / "2.Validation"
    / "\uc6d0\ucc9c\ub370\uc774\ud130"
    / "\uac1c"
    / "\uc548\uad6c"
    / "\uc77c\ubc18"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "project" / "datasets" / "dog_general_abnormal_dataset.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_rows(image_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for disease_dir in sorted([p for p in image_root.iterdir() if p.is_dir()]):
        disease_name = disease_dir.name
        for label_dir in sorted([p for p in disease_dir.iterdir() if p.is_dir()]):
            label_name = label_dir.name
            label_id = 1 if positive_label(label_name) else 0
            for image_path in sorted(label_dir.glob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rows.append(
                    {
                        "image_path": str(image_path),
                        "relative_path": str(image_path.relative_to(PROJECT_ROOT)),
                        "label": str(label_id),
                        "label_name": "abnormal" if label_id else "normal",
                        "source_disease": disease_name,
                        "source_label": label_name,
                    }
                )
    return rows


def assign_splits(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    split_rows: list[dict[str, str]] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        for idx, row in enumerate(label_rows):
            row = dict(row)
            if idx < train_end:
                row["split"] = "train"
            elif idx < val_end:
                row["split"] = "val"
            else:
                row["split"] = "test"
            split_rows.append(row)

    split_rows.sort(key=lambda r: (r["split"], r["label"], r["relative_path"]))
    return split_rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "relative_path",
        "label",
        "label_name",
        "source_disease",
        "source_label",
        "split",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["label"])
        counts[key] = counts.get(key, 0) + 1

    print(f"total: {len(rows)}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {sum(v for (s, _), v in counts.items() if s == split)}")
        print(f"  normal: {counts.get((split, '0'), 0)}")
        print(f"  abnormal: {counts.get((split, '1'), 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.image_root)
    if not rows:
        raise SystemExit(f"No images found under {args.image_root}")

    split_rows = assign_splits(rows, args.seed)
    write_csv(split_rows, args.output)
    print(f"wrote: {args.output}")
    print_summary(split_rows)


if __name__ == "__main__":
    main()
