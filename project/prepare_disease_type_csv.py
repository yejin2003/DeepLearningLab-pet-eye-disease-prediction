from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from disease_definitions import DISEASES, positive_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT
    / "153.\ubc18\ub824\ub3d9\ubb3c \uc548\uad6c\uc9c8\ud658 \ub370\uc774\ud130"
    / "01.\ub370\uc774\ud130"
    / "2.Validation"
    / "\uc6d0\ucc9c\ub370\uc774\ud130"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "project" / "datasets" / "eye_disease_type_dataset.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_TO_ID = {disease_key: idx for idx, disease_key in enumerate(DISEASES)}
ID_TO_CLASS = {idx: disease_key for disease_key, idx in CLASS_TO_ID.items()}


def collect_rows(image_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for disease_key, disease in DISEASES.items():
        disease_name = disease["name_ko"]
        disease_dirs = [p for p in image_root.rglob(disease_name) if p.is_dir()]
        for disease_dir in disease_dirs:
            for label_dir in sorted([p for p in disease_dir.iterdir() if p.is_dir()]):
                if not positive_label(label_dir.name):
                    continue
                for image_path in sorted(label_dir.glob("*")):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    rel_parts = image_path.relative_to(image_root).parts
                    rows.append(
                        {
                            "image_path": str(image_path),
                            "relative_path": str(image_path.relative_to(PROJECT_ROOT)),
                            "class_id": str(CLASS_TO_ID[disease_key]),
                            "class_key": disease_key,
                            "class_name": disease["display_ko"],
                            "source_label": label_dir.name,
                            "species": rel_parts[0] if len(rel_parts) > 0 else "",
                            "device": rel_parts[2] if len(rel_parts) > 2 else "",
                        }
                    )
    return rows


def assign_splits(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["class_key"], []).append(row)

    split_rows: list[dict[str, str]] = []
    for class_rows in by_class.values():
        rng.shuffle(class_rows)
        n = len(class_rows)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        for idx, row in enumerate(class_rows):
            row = dict(row)
            row["split"] = "train" if idx < train_end else "val" if idx < val_end else "test"
            split_rows.append(row)

    return sorted(split_rows, key=lambda r: (r["split"], r["class_id"], r["relative_path"]))


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "relative_path",
        "class_id",
        "class_key",
        "class_name",
        "source_label",
        "species",
        "device",
        "split",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["class_key"])
        counts[key] = counts.get(key, 0) + 1
    print(f"total: {len(rows)}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {sum(v for (s, _), v in counts.items() if s == split)}")
        for class_key in CLASS_TO_ID:
            print(f"  {class_key}: {counts.get((split, class_key), 0)}")


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
        raise SystemExit(f"No disease images found under {args.image_root}")
    split_rows = assign_splits(rows, args.seed)
    write_csv(split_rows, args.output)
    print(f"wrote: {args.output}")
    print_summary(split_rows)


if __name__ == "__main__":
    main()
