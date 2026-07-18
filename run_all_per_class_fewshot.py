from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eval100_protocol import (
    DATASET_CONFIG as EVAL100_DATASET_CONFIG,
    PROTOCOL_TAG,
    validate_eval100_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parent
SHOTS = (1, 3, 5, 10)
DATASETS = {
    "aid": {
        "manifest": EVAL100_DATASET_CONFIG["aid"]["manifest"],
        "examples": EVAL100_DATASET_CONFIG["aid"]["per_class_knn_examples"],
        "config": EVAL100_DATASET_CONFIG["aid"]["config"],
        "data_root": EVAL100_DATASET_CONFIG["aid"]["data_root"],
        "results_prefix": f"aid_{PROTOCOL_TAG}_remoteclip_knn_per_class",
    },
    "nwpu": {
        "manifest": EVAL100_DATASET_CONFIG["nwpu_fg_urban"]["manifest"],
        "examples": EVAL100_DATASET_CONFIG["nwpu_fg_urban"]["per_class_knn_examples"],
        "config": EVAL100_DATASET_CONFIG["nwpu_fg_urban"]["config"],
        "data_root": EVAL100_DATASET_CONFIG["nwpu_fg_urban"]["data_root"],
        "results_prefix": f"nwpu_{PROTOCOL_TAG}_remoteclip_knn_per_class",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate class-balanced RemoteCLIP KNN examples and run frozen-backbone "
            "ImageNet few-shot baselines for AID and NWPU-FG-Urban."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet18", "resnet50", "vit_tiny", "vit_small"],
        default=["resnet18", "resnet50", "vit_tiny", "vit_small"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default=None)
    parser.add_argument(
        "--results-root",
        default=f"results_{PROTOCOL_TAG}/traditional_per_class_fewshot",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Reuse existing per-class example CSV files and run only training.",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Generate and validate example CSV files without training models.",
    )
    parser.add_argument(
        "--rebuild-manifests",
        action="store_true",
        help="Regenerate the fixed 100-images-per-class manifests before retrieval.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def display(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)


def run(command: list[str], dry_run: bool) -> None:
    display(command)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)


def require_file(relative_path: str) -> None:
    path = PROJECT_DIR / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def prepare_manifest(dataset: str, config: dict, args) -> None:
    command = [
        sys.executable,
        "prepare_manifest.py",
        "--config",
        config["config"],
        "--data-root",
        config["data_root"],
        "--out-dir",
        config["manifest"],
        "--eval-per-class",
        "100",
        "--seed",
        str(args.seed),
    ]
    print(f"\n[{dataset.upper()}] Preparing eval100 manifest", flush=True)
    run(command, args.dry_run)


def build_examples(dataset: str, config: dict, args) -> None:
    command = [
        sys.executable,
        "build_examples.py",
        "--manifest-dir",
        config["manifest"],
        "--strategy",
        "knn",
        "--knn-scope",
        "per_class",
        "--shots",
        *[str(shot) for shot in args.shots],
        "--out-dir",
        config["examples"],
        "--feature-backend",
        "remoteclip",
        "--remoteclip-cache",
        args.remoteclip_cache,
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--feature-num-workers",
        str(args.feature_num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.remoteclip_checkpoint:
        command.extend(["--remoteclip-checkpoint", args.remoteclip_checkpoint])
    print(f"\n[{dataset.upper()}] Generating per-class RemoteCLIP examples", flush=True)
    run(command, args.dry_run)


def train(dataset: str, config: dict, shot: int, args) -> None:
    examples_csv = f"{config['examples']}/examples_knn_shot_{shot}.csv"
    if not args.dry_run:
        require_file(examples_csv)
    output_dir = f"{args.results_root}/{config['results_prefix']}_shot_{shot}_head"
    command = [
        sys.executable,
        "run_strict_baselines.py",
        "--manifest-dir",
        config["manifest"],
        "--examples-csv",
        examples_csv,
        "--models",
        *args.models,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--train-mode",
        "head",
        "--seed",
        str(args.seed),
        "--out-dir",
        output_dir,
    ]
    print(f"\n[{dataset.upper()}] Training per-class {shot}-shot baselines", flush=True)
    run(command, args.dry_run)


def main() -> None:
    args = parse_args()
    if args.skip_examples and args.examples_only:
        raise SystemExit("--skip-examples and --examples-only cannot be used together.")

    for dataset in args.datasets:
        config = DATASETS[dataset]
        evaluation_csv = PROJECT_DIR / config["manifest"] / "evaluation.csv"
        if args.rebuild_manifests or not evaluation_csv.is_file():
            prepare_manifest(dataset, config, args)
        if not args.dry_run:
            require_file(f"{config['manifest']}/evaluation.csv")
            require_file(f"{config['manifest']}/support.csv")
            require_file(f"{config['manifest']}/class_order.json")
            require_file(f"{config['manifest']}/summary.json")
            validate_eval100_manifest(PROJECT_DIR / config["manifest"])
        if not args.skip_examples:
            build_examples(dataset, config, args)

    if args.examples_only:
        print("\nExample generation completed.", flush=True)
        return

    for dataset in args.datasets:
        config = DATASETS[dataset]
        for shot in args.shots:
            train(dataset, config, shot, args)

    print("\nAll requested few-shot experiments completed.", flush=True)


if __name__ == "__main__":
    main()
