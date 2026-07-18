from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from eval100_protocol import (
    DATASET_CONFIG,
    EVAL_PER_CLASS,
    PROTOCOL_TAG,
    SEED,
    validate_eval100_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Prepare manifests and example CSVs for {PROTOCOL_TAG}."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIG),
        default=sorted(DATASET_CONFIG),
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["random", "knn"],
        default=["random", "knn"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--skip-manifests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("\n> " + " ".join(repr(part) if " " in part else part for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)

    for dataset in args.datasets:
        cfg = DATASET_CONFIG[dataset]
        if not args.skip_manifests:
            run([
                args.python,
                str(root / "prepare_manifest.py"),
                "--config", cfg["config"],
                "--data-root", cfg["data_root"],
                "--out-dir", cfg["manifest"],
                "--eval-per-class", str(EVAL_PER_CLASS),
                "--seed", str(SEED),
            ], args.dry_run)

        if not args.dry_run:
            validate_eval100_manifest(root / cfg["manifest"])

        if "random" in args.strategies:
            run([
                args.python,
                str(root / "build_examples.py"),
                "--manifest-dir", cfg["manifest"],
                "--strategy", "random",
                "--shots", *map(str, args.shots),
                "--out-dir", cfg["random_examples"],
                "--seed", str(SEED),
            ], args.dry_run)

        if "knn" in args.strategies:
            command = [
                args.python,
                str(root / "build_examples.py"),
                "--manifest-dir", cfg["manifest"],
                "--strategy", "knn",
                "--shots", *map(str, args.shots),
                "--out-dir", cfg["knn_examples"],
                "--feature-backend", "remoteclip",
                "--remoteclip-cache", args.remoteclip_cache,
                "--feature-batch-size", str(args.feature_batch_size),
                "--feature-num-workers", str(args.feature_num_workers),
            ]
            if args.remoteclip_checkpoint:
                command.extend(["--remoteclip-checkpoint", args.remoteclip_checkpoint])
            run(command, args.dry_run)


if __name__ == "__main__":
    main()
