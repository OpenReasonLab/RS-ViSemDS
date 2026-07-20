from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROTOCOL_TAG = "eval100_seed42"
EVAL_PER_CLASS = 100
SEED = 42

DATASET_CONFIG = {
    "aid": {
        "manifest": "manifests/aid_eval100_seed42",
        "random_examples": "examples/aid_eval100_random_seed42",
        "knn_examples": "examples/aid_eval100_remoteclip_knn",
        "per_class_knn_examples_root": "examples/aid_eval100_backbone_knn_per_class",
        "config": "configs/aid.json",
        "data_root": "data_raw/AID_dataset",
    },
    "nwpu_fg_urban": {
        "manifest": "manifests/nwpu_eval100_seed42",
        "random_examples": "examples/nwpu_eval100_random_seed42",
        "knn_examples": "examples/nwpu_eval100_remoteclip_knn",
        "per_class_knn_examples_root": "examples/nwpu_eval100_backbone_knn_per_class",
        "config": "configs/nwpu_fg_urban.json",
        "data_root": "data_raw/NWPU-RESISC45",
    },
}


def validate_eval100_manifest(manifest_dir: Path) -> None:
    evaluation = _read_csv(manifest_dir / "evaluation.csv")
    support = _read_csv(manifest_dir / "support.csv")
    class_order = _read_json(manifest_dir / "class_order.json")["classes"]
    summary = _read_json(manifest_dir / "summary.json")

    counts = Counter(row["label"] for row in evaluation)
    expected = {label: EVAL_PER_CLASS for label in class_order}
    if dict(counts) != expected:
        raise ValueError(
            f"Invalid {PROTOCOL_TAG} manifest at {manifest_dir}: "
            f"expected evaluation counts {expected}, got {dict(counts)}"
        )

    if summary.get("eval_per_class") != EVAL_PER_CLASS:
        raise ValueError(
            f"Manifest {manifest_dir} records eval_per_class="
            f"{summary.get('eval_per_class')!r}, expected {EVAL_PER_CLASS}."
        )
    if summary.get("seed") != SEED:
        raise ValueError(
            f"Manifest {manifest_dir} records seed={summary.get('seed')!r}, "
            f"expected {SEED}."
        )

    evaluation_paths = {row["path"] for row in evaluation}
    support_paths = {row["path"] for row in support}
    overlap = evaluation_paths & support_paths
    if overlap:
        raise ValueError(
            f"Evaluation/support leakage detected in {manifest_dir}: "
            f"{sorted(overlap)[:5]}"
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
