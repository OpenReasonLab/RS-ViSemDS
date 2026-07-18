from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval100_protocol import DATASET_CONFIG, validate_eval100_manifest


MODEL_DEFAULTS = {
    "gemma3_12b": os.environ.get("GEMMA3_12B_MODEL", "/root/autodl-tmp/models/gemma-3-12b-it"),
    "qwen3vl_8b": os.environ.get("QWEN3VL_8B_MODEL", "/root/autodl-tmp/models/Qwen3-VL-8B"),
    "internvl35_14b": os.environ.get("INTERNVL35_14B_MODEL", "/root/autodl-tmp/models/InternVL3.5-14B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main RS-ViSemDS eval100 suite.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIG), default=sorted(DATASET_CONFIG))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_DEFAULTS), default=list(MODEL_DEFAULTS))
    parser.add_argument("--model-path", action="append", default=[], help="Override as alias=/path/to/model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths = dict(MODEL_DEFAULTS)
    for item in args.model_path:
        if "=" not in item:
            raise ValueError("--model-path must use alias=/path syntax")
        alias, path = item.split("=", 1)
        if alias not in MODEL_DEFAULTS:
            raise ValueError(f"Unknown model alias: {alias}")
        model_paths[alias] = path

    for dataset in args.datasets:
        cfg = DATASET_CONFIG[dataset]
        manifest = PROJECT_ROOT / cfg["manifest"]
        if not args.dry_run:
            validate_eval100_manifest(manifest)
        selection_rel = Path("RS-ViSemDS") / "examples" / f"{dataset}_eval100_seed42"
        selection_dir = PROJECT_ROOT / selection_rel
        selected_csv_rel = selection_rel / f"examples_rs_visemds_shot_{args.k}.csv"
        selected_csv = PROJECT_ROOT / selected_csv_rel
        if args.rebuild_selection or not selected_csv.exists():
            command = [
                args.python, str(SCRIPT_ROOT / "build_rs_visemds_examples.py"),
                "--dataset", dataset,
                "--manifest-dir", cfg["manifest"],
                "--out-dir", str(selection_rel),
                "--r", str(args.r), "--k", str(args.k),
                "--remoteclip-cache", args.remoteclip_cache,
                "--feature-batch-size", str(args.feature_batch_size),
                "--feature-num-workers", str(args.feature_num_workers),
            ]
            if args.remoteclip_checkpoint:
                command.extend(["--remoteclip-checkpoint", args.remoteclip_checkpoint])
            run(command, args.dry_run)

        for alias in args.models:
            model_path = model_paths[alias]
            if not args.dry_run and not Path(model_path).is_dir():
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            out_dir_rel = Path("RS-ViSemDS") / "results_eval100_seed42" / f"{alias}_{dataset}"
            command = [
                args.python, str(SCRIPT_ROOT / "run_rs_visemds_mllm.py"),
                "--dataset", dataset,
                "--manifest-dir", cfg["manifest"],
                "--selected-examples-csv", str(selected_csv_rel),
                "--model", model_path,
                "--out-dir", str(out_dir_rel),
                "--max-tokens", str(args.max_tokens),
                "--resume",
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            run(command, args.dry_run)


def run(command: list[str], dry_run: bool) -> None:
    print("\n> " + " ".join(repr(part) if " " in part else part for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
