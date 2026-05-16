#!/usr/bin/env python3
"""Inspect Fashionpedia category list from a COCO-style annotation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print all categories (id, name) from Fashionpedia annotations JSON."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to instances_attributes_train2020.json or instances_attributes_val2020.json",
    )
    return parser.parse_args()


def load_categories(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format in {json_path}: {exc}") from exc

    categories = data.get("categories")
    if categories is None:
        raise KeyError("Field 'categories' not found in JSON")
    if not isinstance(categories, list):
        raise TypeError("Field 'categories' must be a list")

    return categories


def main() -> None:
    args = parse_args()

    try:
        categories = load_categories(args.json_path)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)

    # Filter only valid entries and sort for stable output.
    parsed = []
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        cid = cat.get("id")
        name = cat.get("name")
        if cid is None or name is None:
            continue
        parsed.append((cid, str(name)))

    parsed.sort(key=lambda x: x[0])

    print(f"Found {len(parsed)} categories in: {args.json_path}")
    print("-" * 60)
    for cid, name in parsed:
        print(f"id={cid:>3} | name={name}")


if __name__ == "__main__":
    main()
