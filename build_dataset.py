#!/usr/bin/env python3
"""Build dataset_final from Kaggle Fashion Product Images styles metadata."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(r"C:\Users\zudae\Desktop\диплом1\model3\fashion-dataset")
OUTPUT_DIRNAME = "dataset_final"
SEED = 42
TRAIN_RATIO = 0.8

PROGRAM_CLASSES = [
    "tshirt",
    "top",
    "shirt",
    "longsleeve",
    "sweater",
    "hoodie",
    "cardigan",
    "vest",
    "blazer",
    "jacket",
    "leatherJacket",
    "windbreaker",
    "coat",
    "trench",
    "downJacket",
    "raincoat",
    "jeans",
    "trousers",
    "skirt",
    "shorts",
    "sneakers",
    "boots",
    "highBoots",
    "shoes",
    "sandals",
    "hat",
    "cap",
    "scarf",
    "umbrella",
    "dress",
]

ARTICLE_TYPE_MAPPING = {
    "Tshirts": "tshirt",
    "Tops": "top",
    "Shirts": "shirt",
    "Sweatshirts": "hoodie",
    "Sweaters": "sweater",
    "Blazers": "blazer",
    "Jackets": "jacket",
    "Nehru Jackets": "jacket",
    "Rain Jacket": "raincoat",
    "Waistcoat": "vest",
    "Innerwear Vests": "vest",
    "Jeans": "jeans",
    "Jeggings": "jeans",
    "Trousers": "trousers",
    "Track Pants": "trousers",
    "Capris": "trousers",
    "Lounge Pants": "trousers",
    "Rain Trousers": "trousers",
    "Leggings": "trousers",
    "Skirts": "skirt",
    "Shorts": "shorts",
    "Lounge Shorts": "shorts",
    "Sports Shoes": "sneakers",
    "Casual Shoes": "shoes",
    "Formal Shoes": "shoes",
    "Heels": "shoes",
    "Flats": "shoes",
    "Booties": "boots",
    "Sandals": "sandals",
    "Sports Sandals": "sandals",
    "Flip Flops": "sandals",
    "Caps": "cap",
    "Hat": "hat",
    "Scarves": "scarf",
    "Mufflers": "scarf",
    "Stoles": "scarf",
    "Umbrellas": "umbrella",
    "Dresses": "dress",
    "Jumpsuit": "dress",
    "Rompers": "dress",
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


def reset_output_dirs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val"):
        for class_name in PROGRAM_CLASSES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)


def split_counts(total: int) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    if total == 1:
        return 1, 0
    if 2 <= total <= 4:
        val_count = 1
        return total - val_count, val_count

    train_count = int(total * TRAIN_RATIO)
    val_count = total - train_count
    if val_count == 0:
        val_count = 1
        train_count = total - 1
    return train_count, val_count


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
    class_to_source_types: dict[str, set[str]] = defaultdict(set)
    unmapped_counter: Counter[str] = Counter()

    found_images = 0
    missing_images = 0

    for _, row in df.iterrows():
        article_type_raw = row.get("articleType")
        item_id = row.get("id")

        if pd.isna(article_type_raw) or pd.isna(item_id):
            continue

        article_type = str(article_type_raw).strip()
        if not article_type:
            continue

        if article_type not in ARTICLE_TYPE_MAPPING:
            unmapped_counter[article_type] += 1
            continue

        try:
            image_name = f"{int(item_id)}.jpg"
        except (TypeError, ValueError):
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            missing_images += 1
            continue

        found_images += 1
        class_name = ARTICLE_TYPE_MAPPING[article_type]
        class_to_images[class_name].append(image_path)
        class_to_source_types[class_name].add(article_type)

    reset_output_dirs(output_root)
    random.seed(SEED)

    stats_rows: list[dict[str, object]] = []
    total_copied = 0
    empty_classes: list[str] = []

    for class_name in PROGRAM_CLASSES:
        images = class_to_images.get(class_name, [])
        images = images.copy()
        random.shuffle(images)

        total_available = len(images)
        train_count, val_count = split_counts(total_available)

        train_images = images[:train_count]
        val_images = images[train_count : train_count + val_count]

        for src in train_images:
            shutil.copy2(src, output_root / "train" / class_name / src.name)
        for src in val_images:
            shutil.copy2(src, output_root / "val" / class_name / src.name)

        total_copied += len(train_images) + len(val_images)

        source_types = sorted(class_to_source_types.get(class_name, set()))
        status = "used" if total_available > 0 else "empty_no_matching_article_type"
        if total_available == 0:
            empty_classes.append(class_name)

        stats_rows.append(
            {
                "class_name": class_name,
                "total_available": total_available,
                "train_count": len(train_images),
                "val_count": len(val_images),
                "source_article_types": "|".join(source_types),
                "status": status,
            }
        )

    (output_root / "class_names.json").write_text(
        json.dumps(PROGRAM_CLASSES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(stats_rows).to_csv(output_root / "dataset_stats.csv", index=False, encoding="utf-8")

    unmapped_rows = [
        {"articleType": article_type, "count": count}
        for article_type, count in sorted(unmapped_counter.items(), key=lambda x: (-x[1], x[0]))
    ]
    pd.DataFrame(unmapped_rows, columns=["articleType", "count"]).to_csv(
        output_root / "unmapped_article_types.csv", index=False, encoding="utf-8"
    )

    print("=" * 80)
    print(f"Total rows in styles.csv: {total_rows}")
    print(f"Images found: {found_images}")
    print(f"Images not found: {missing_images}")
    print(f"Total copied images: {total_copied}")
    print("-" * 80)
    print("Per-class stats:")
    for row in stats_rows:
        print(
            f"  {row['class_name']}: total={row['total_available']}, "
            f"train={row['train_count']}, val={row['val_count']}, status={row['status']}"
        )
    print("-" * 80)
    print("Empty classes:")
    if empty_classes:
        for class_name in empty_classes:
            print(f"  {class_name}")
    else:
        print("  None")
    print("-" * 80)
    print(f"Output dataset path: {output_root}")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    try:
        build_dataset(args.root)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
