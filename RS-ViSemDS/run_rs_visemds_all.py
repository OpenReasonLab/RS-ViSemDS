from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from eval100_protocol import DATASET_CONFIG, validate_eval100_manifest
from strict_fewshot.utils import sha256_file
from rs_visemds.prompt_builder import PROMPT_MODES


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
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="manuscript_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = (args.alpha, args.beta, args.gamma)
    if any(value < 0 for value in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("alpha, beta, and gamma must be non-negative and sum to 1")
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
        weight_tag = _weight_tag(args.alpha, args.beta, args.gamma)
        selection_rel = (
            Path("RS-ViSemDS") / "examples" /
            f"{dataset}_eval100_seed42_{weight_tag}"
        )
        selection_dir = PROJECT_ROOT / selection_rel
        selected_csv_rel = selection_rel / f"examples_rs_visemds_shot_{args.k}.csv"
        selected_csv = PROJECT_ROOT / selected_csv_rel
        selection_config = selection_dir / "selection_config.json"
        compatible = _selection_is_compatible(
            selection_config=selection_config,
            selected_csv=selected_csv,
            manifest=manifest,
            dataset=dataset,
            r=args.r,
            k=args.k,
            weights=(args.alpha, args.beta, args.gamma),
        )
        if selected_csv.exists() and not compatible and not args.rebuild_selection and not args.dry_run:
            raise ValueError(
                f"Existing RS-ViSemDS selection is incompatible: {selection_dir}. "
                "Use --rebuild-selection after verifying the requested weights."
            )
        if args.rebuild_selection or not compatible:
            command = [
                args.python, str(SCRIPT_ROOT / "build_rs_visemds_examples.py"),
                "--dataset", dataset,
                "--manifest-dir", cfg["manifest"],
                "--out-dir", str(selection_rel),
                "--r", str(args.r), "--k", str(args.k),
                "--alpha", str(args.alpha),
                "--beta", str(args.beta),
                "--gamma", str(args.gamma),
                "--remoteclip-cache", args.remoteclip_cache,
                "--feature-batch-size", str(args.feature_batch_size),
                "--feature-num-workers", str(args.feature_num_workers),
            ]
            if args.remoteclip_checkpoint:
                command.extend(["--remoteclip-checkpoint", args.remoteclip_checkpoint])
            run(command, args.dry_run)
            if not args.dry_run and not _selection_is_compatible(
                selection_config=selection_config,
                selected_csv=selected_csv,
                manifest=manifest,
                dataset=dataset,
                r=args.r,
                k=args.k,
                weights=(args.alpha, args.beta, args.gamma),
            ):
                raise RuntimeError(f"Generated selection failed compatibility checks: {selection_dir}")

        for alias in args.models:
            model_path = model_paths[alias]
            if not args.dry_run and not Path(model_path).is_dir():
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            out_dir_rel = (
                Path("RS-ViSemDS") / "results_eval100_seed42" / weight_tag /
                args.prompt_mode /
                f"{alias}_{dataset}"
            )
            command = [
                args.python, str(SCRIPT_ROOT / "run_rs_visemds_mllm.py"),
                "--dataset", dataset,
                "--manifest-dir", cfg["manifest"],
                "--selected-examples-csv", str(selected_csv_rel),
                "--model", model_path,
                "--out-dir", str(out_dir_rel),
                "--max-tokens", str(args.max_tokens),
                "--prompt-mode", args.prompt_mode,
                "--resume",
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            run(command, args.dry_run)


def run(command: list[str], dry_run: bool) -> None:
    print("\n> " + " ".join(repr(part) if " " in part else part for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _weight_tag(alpha: float, beta: float, gamma: float) -> str:
    def component(value: float) -> str:
        return f"{value:.6g}".replace("-", "m").replace(".", "p")

    return f"a{component(alpha)}_b{component(beta)}_g{component(gamma)}"


def _selection_is_compatible(
    selection_config: Path,
    selected_csv: Path,
    manifest: Path,
    dataset: str,
    r: int,
    k: int,
    weights: tuple[float, float, float],
) -> bool:
    if not selection_config.is_file() or not selected_csv.is_file():
        return False
    try:
        with selection_config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        stored = config["weights"]
        stored_weights = (
            float(stored["alpha"]),
            float(stored["beta"]),
            float(stored["gamma"]),
        )
        return (
            config.get("method") == "RS-ViSemDS"
            and config.get("dataset") == dataset
            and int(config.get("r_per_class")) == r
            and int(config.get("k_total_demonstrations")) == k
            and all(abs(actual - expected) <= 1e-12 for actual, expected in zip(stored_weights, weights))
            and config.get("evaluation_sha256") == sha256_file(manifest / "evaluation.csv")
            and config.get("support_sha256") == sha256_file(manifest / "support.csv")
            and config.get("class_order_sha256") == sha256_file(manifest / "class_order.json")
            and config.get("selected_examples_sha256") == sha256_file(selected_csv)
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


if __name__ == "__main__":
    main()
