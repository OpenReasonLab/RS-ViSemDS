from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path

from strict_fewshot.utils import list_images, read_json, repo_path, stable_id, write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--eval-per-class", type=int, default=None)
    split_group.add_argument("--eval-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def normalize_dir_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_dataset_root(data_root: Path, classes: list[str]) -> tuple[Path, dict[str, Path]]:
    expected = {normalize_dir_name(label): label for label in classes}
    candidates = [data_root]
    frontier = [data_root]

    for _ in range(3):
        next_frontier = []
        for root in frontier:
            if not root.exists() or not root.is_dir():
                continue
            next_frontier.extend(path for path in root.iterdir() if path.is_dir())
        candidates.extend(next_frontier)
        frontier = next_frontier

    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue
        child_dirs = {
            normalize_dir_name(path.name): path
            for path in root.iterdir()
            if path.is_dir()
        }
        if all(name in child_dirs for name in expected):
            return root, {
                label: child_dirs[normalize_dir_name(label)]
                for label in classes
            }

    expected_names = ", ".join(classes)
    raise FileNotFoundError(
        f"Could not find a dataset directory containing all configured classes under {data_root}. "
        f"Expected: {expected_names}"
    )


def main():
    args = parse_args()
    base = Path.cwd()
    config_path = repo_path(args.config, base)
    data_root = repo_path(args.data_root, base)
    out_dir = repo_path(args.out_dir, base)
    config = read_json(config_path)

    eval_per_class = args.eval_per_class
    eval_ratio = args.eval_ratio
    if eval_per_class is None and eval_ratio is None:
        eval_per_class = 24
    if eval_ratio is not None and not 0.0 < eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be between 0 and 1.")

    classes = config["classes"]
    max_shot = int(config.get("max_shot", 10))
    expected_per_class = config.get("expected_images_per_class")
    dataset_root, class_dirs = find_dataset_root(data_root, classes)
    relative_dataset_root = Path(os.path.relpath(dataset_root, base)).as_posix()
    evaluation_rows = []
    support_rows = []
    summary = {
        "dataset": config["dataset"],
        "data_root": relative_dataset_root,
        "requested_data_root": args.data_root,
        "split_mode": "stratified_ratio" if eval_ratio is not None else "fixed_per_class",
        "eval_per_class": eval_per_class,
        "eval_ratio": eval_ratio,
        "evaluation_role": "validation" if eval_ratio is not None else "fixed_evaluation",
        "seed": args.seed,
        "classes": classes,
        "class_counts": {},
    }

    for label in classes:
        class_dir = class_dirs[label]
        images = list_images(class_dir)
        if (
            expected_per_class is not None
            and len(images) != int(expected_per_class)
            and not args.allow_incomplete
        ):
            raise ValueError(
                f"Class {label} has {len(images)} images, expected exactly "
                f"{expected_per_class}. Use the complete original dataset, or pass "
                "--allow-incomplete only for pipeline debugging."
            )
        class_eval_count = (
            int(round(len(images) * eval_ratio))
            if eval_ratio is not None
            else int(eval_per_class)
        )
        required = class_eval_count + 1
        if len(images) < required:
            raise ValueError(
                f"Class {label} has {len(images)} images, but strict protocol needs "
                f"at least evaluation_count + 1 training image = {required}."
            )

        rng = random.Random(args.seed + classes.index(label))
        shuffled = images[:]
        rng.shuffle(shuffled)
        eval_imgs = sorted(shuffled[:class_eval_count])
        support_imgs = sorted(shuffled[class_eval_count:])

        for path in eval_imgs:
            rel = path.relative_to(dataset_root).as_posix()
            evaluation_rows.append({
                "target_id": stable_id(label, rel),
                "label": label,
                "path": rel,
            })
        for idx, path in enumerate(support_imgs):
            rel = path.relative_to(dataset_root).as_posix()
            support_rows.append({
                "sample_id": f"{label}__support_{idx:05d}",
                "label": label,
                "path": rel,
            })

        summary["class_counts"][label] = {
            "source_directory": class_dir.name,
            "total": len(images),
            "evaluation": len(eval_imgs),
            "support": len(support_imgs),
            "validation_ratio": len(eval_imgs) / len(images),
            "training_ratio": len(support_imgs) / len(images),
        }

    if len(support_rows) < max_shot:
        raise ValueError(
            f"Support pool has {len(support_rows)} images, but max_shot={max_shot}."
        )

    write_csv(out_dir / "evaluation.csv", evaluation_rows, ["target_id", "label", "path"])
    write_csv(out_dir / "support.csv", support_rows, ["sample_id", "label", "path"])
    write_json(out_dir / "class_order.json", {"classes": classes})
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote manifest to {out_dir}")


if __name__ == "__main__":
    main()
