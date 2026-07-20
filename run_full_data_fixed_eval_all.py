from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "aid": {
        "manifest": "manifests/aid_eval100_seed42",
        "output": "results_eval100_seed42/full_data_fixed_epoch10/aid",
    },
    "nwpu": {
        "manifest": "manifests/nwpu_eval100_seed42",
        "output": "results_eval100_seed42/full_data_fixed_epoch10/nwpu_urban",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four fixed-evaluation full-data baselines on AID and NWPU-Urban."
    )
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet18", "resnet50", "vit_tiny", "vit_small"],
        default=["resnet18", "resnet50", "vit_tiny", "vit_small"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-epochs", type=int, choices=[10], default=10)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-images", type=int, default=20)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    runner = project_root / "run_full_data_fixed_eval.py"
    for dataset_name in args.datasets:
        setting = DATASETS[dataset_name]
        command = [
            sys.executable,
            "-u",
            str(runner),
            "--manifest-dir",
            setting["manifest"],
            "--out-dir",
            setting["output"],
            "--models",
            *args.models,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--max-epochs",
            str(args.max_epochs),
            "--validation-ratio",
            str(args.validation_ratio),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--num-workers",
            str(args.num_workers),
            "--warmup-images",
            str(args.warmup_images),
        ]
        if args.limit_per_class:
            command.extend(["--limit-per-class", str(args.limit_per_class)])
        if args.resume:
            command.append("--resume")
        print(f"Starting {dataset_name}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=project_root, check=True)
    print("All requested datasets completed.", flush=True)


if __name__ == "__main__":
    main()
