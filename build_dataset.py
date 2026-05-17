#!/usr/bin/env python3
"""Build train/val image classification dataset from fashion-dataset styles.csv."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(r"C:\Users\zudae\Desktop\диплом1\model3\fashion-dataset")
OUTPUT_DIRNAME = "dataset_final"
SEED = 42
TRAIN_RATIO = 0.8
MAX_TRAIN_PER_CLASS = 700
MAX_VAL_PER_CLASS = 150

ARTICLE_TYPE_MAPPING = {
    "Tshirts": "tshirt",
    "Tops": "top",
    "Shirts": "shirt",
    "Sweatshirts": "hoodie",
    "Sweaters": "sweater",
    "Blazers": "blazer",
    "Jackets": "jacket",
    "Rain Jacket": "raincoat",
    "Jeans": "jeans",
    "Trousers": "trousers",
    "Skirts": "skirt",
    "Shorts": "shorts",
    "Sports Shoes": "sneakers",
    "Casual Shoes": "shoes",
    "Formal Shoes": "shoes",
    "Booties": "boots",
    "Sandals": "sandals",
    "Dresses": "dress",
    "Caps": "cap",
    "Hat": "hat",
    "Scarves": "scarf",
    "Umbrellas": "umbrella",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset_final from fashion-dataset metadata")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Path to fashion-dataset root (default: {DEFAULT_ROOT})",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path, class_names: list[str]) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        (output_root / "train" / class_name).mkdir(parents=True, exist_ok=True)
        (output_root / "val" / class_name).mkdir(parents=True, exist_ok=True)


def build_dataset(root: Path) -> None:
    styles_csv = root / "styles.csv"
    images_dir = root / "images"
    output_root = root / OUTPUT_DIRNAME

    if not styles_csv.exists():
        raise FileNotFoundError(f"styles.csv not found: {styles_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images folder not found: {images_dir}")

    df = pd.read_csv(styles_csv, on_bad_lines="skip")
    total_rows = len(df)

    class_to_images: dict[str, list[Path]] = defaultdict(list)
    matched_rows = 0
    missing_files = 0

    for _, row in df.iterrows():
        article_type = row.get("articleType")
        item_id = row.get("id")

        if pd.isna(article_type) or pd.isna(item_id):
            continue

        article_type = str(article_type).strip()
        if not article_type:
            continue

        if article_type not in ARTICLE_TYPE_MAPPING:
            continue

        matched_rows += 1
        class_name = ARTICLE_TYPE_MAPPING[article_type]

        try:
            image_name = f"{int(item_id)}.jpg"
        except (TypeError, ValueError):
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            missing_files += 1
            continue

        class_to_images[class_name].append(image_path)

    class_names = sorted(class_to_images.keys())
    ensure_dirs(output_root, class_names)

    random.seed(SEED)

    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}
    total_available_counts: dict[str, int] = {}

    for class_name in class_names:
        images = class_to_images[class_name]
        random.shuffle(images)

        total_available = len(images)
        split_idx = int(total_available * TRAIN_RATIO)
        train_images = images[:split_idx][:MAX_TRAIN_PER_CLASS]
        val_images = images[split_idx:][:MAX_VAL_PER_CLASS]

        total_available_counts[class_name] = total_available
        train_counts[class_name] = len(train_images)
        val_counts[class_name] = len(val_images)

        for src in train_images:
            dst = output_root / "train" / class_name / src.name
            shutil.copy2(src, dst)

        for src in val_images:
            dst = output_root / "val" / class_name / src.name
            shutil.copy2(src, dst)

    (output_root / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats_df = pd.DataFrame(
        [
            {
                "class_name": class_name,
                "total_available": total_available_counts.get(class_name, 0),
                "train_count": train_counts.get(class_name, 0),
                "val_count": val_counts.get(class_name, 0),
            }
            for class_name in class_names
        ]
    )
    stats_df.to_csv(output_root / "dataset_stats.csv", index=False, encoding="utf-8")

    low_data_classes = [
        class_name
        for class_name in class_names
        if total_available_counts.get(class_name, 0) < 100
    ]

    print("=" * 70)
    print(f"Total rows in styles.csv: {total_rows}")
    print(f"Rows matched by mapping: {matched_rows}")
    print(f"Image files not found: {missing_files}")
    print("-" * 70)
    print("Copied to train per class:")
    for class_name in class_names:
        print(f"  {class_name}: {train_counts[class_name]}")
    print("-" * 70)
    print("Copied to val per class:")
    for class_name in class_names:
        print(f"  {class_name}: {val_counts[class_name]}")
    print("-" * 70)
    print("Classes with fewer than 100 available images:")
    if low_data_classes:
        for class_name in low_data_classes:
            print(f"  {class_name} ({total_available_counts[class_name]})")
    else:
        print("  None")
    print("=" * 70)
    print(f"Output dataset created at: {output_root}")


def main() -> None:
    args = parse_args()
    try:
        build_dataset(args.root)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
