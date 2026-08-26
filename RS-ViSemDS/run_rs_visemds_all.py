from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent


MODEL_DEFAULTS = {
    "gemma3_12b": os.environ.get(
        "GEMMA3_12B_MODEL", "/root/autodl-tmp/models/gemma-3-12b-it"
    ),
    "qwen3vl_8b": os.environ.get(
        "QWEN3VL_8B_MODEL", "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
    ),
    "internvl35_14b": os.environ.get(
        "INTERNVL35_14B_MODEL", "/root/autodl-tmp/models/InternVL3.5-14B"
    ),
}

DATASETS = {
    "aid": "aid",
    "nwpu_urban": "nwpu_urban",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-version adaptive RS-ViSemDS main suite."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_DEFAULTS, default=list(MODEL_DEFAULTS)
    )
    parser.add_argument(
        "--model-path", action="append", default=[], help="Override as alias=/model/path"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out-root", default="RS-ViSemDS/results_paper_v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--calibration-samples-per-class", type=int, default=50)
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs != 10:
        raise ValueError("The formal paper protocol requires exactly 10 runs")
    model_paths = dict(MODEL_DEFAULTS)
    for value in args.model_path:
        if "=" not in value:
            raise ValueError("--model-path must use alias=/path syntax")
        alias, path = value.split("=", 1)
        if alias not in model_paths:
            raise ValueError(f"Unknown model alias: {alias}")
        model_paths[alias] = path

    for dataset in args.datasets:
        for alias in args.models:
            model_path = model_paths[alias]
            if not args.dry_run and not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            run_dirs: list[Path] = []
            for run_index in range(1, args.runs + 1):
                run_dir = Path(args.out_root) / alias / dataset / f"run_{run_index:02d}"
                run_dirs.append(run_dir)
                command = [
                    args.python,
                    str(SCRIPT_ROOT / "run_rs_visemds.py"),
                    "--dataset", DATASETS[dataset],
                    "--weight-mode", "adaptive",
                    "--model-id", model_path,
                    "--output-dir", str(run_dir),
                    "--calibration-seed", str(args.calibration_seed),
                    "--calibration-samples-per-class",
                    str(args.calibration_samples_per_class),
                    "--remoteclip-cache", args.remoteclip_cache,
                    "--feature-batch-size", str(args.feature_batch_size),
                    "--feature-num-workers", str(args.feature_num_workers),
                    "--max-tokens", "256",
                    "--resume",
                ]
                if args.remoteclip_checkpoint:
                    command.extend(["--remoteclip-checkpoint", args.remoteclip_checkpoint])
                if args.limit is not None:
                    command.extend(["--limit", str(args.limit), "--diagnostic-only"])
                run(command, args.dry_run)
            if args.limit is None:
                aggregate_path = (
                    Path(args.out_root) / alias / dataset / "ten_run_summary.json"
                )
                run(
                    [
                        args.python,
                        str(PROJECT_ROOT / "aggregate_paper_runs.py"),
                        *[str(path) for path in run_dirs],
                        "--output", str(aggregate_path),
                    ],
                    args.dry_run,
                )


def run(command: list[str], dry_run: bool) -> None:
    print("\n> " + " ".join(repr(part) if " " in part else part for part in command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
