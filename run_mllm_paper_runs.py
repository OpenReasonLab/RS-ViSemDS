from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from eval100_protocol import DATASET_CONFIG, validate_eval100_manifest


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS = 10
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manuscript MLLM baselines with ten-run aggregation."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIG),
        default=sorted(DATASET_CONFIG),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("zero", "random", "knn"),
        default=["zero", "random", "knn"],
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[3])
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--backend", choices=("transformers", "api"), default="transformers")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--out-root", default="results_paper_v1/mllm_baselines")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-knn", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs != RUNS:
        raise ValueError("The manuscript protocol requires exactly ten runs")
    if args.seed != SEED:
        raise ValueError("The manuscript sampling protocol starts from seed 42")
    if not args.shots or any(shot <= 0 for shot in args.shots):
        raise ValueError("--shots must contain positive integers")
    if args.backend == "api" and not args.api_key and not args.dry_run:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key for the official API")
    return args


def display_command(command: list[str]) -> str:
    displayed = list(command)
    for index, value in enumerate(displayed[:-1]):
        if value == "--api-key":
            displayed[index + 1] = "***REDACTED***"
    return " ".join(repr(value) if " " in value else value for value in displayed)


def run(command: list[str], *, dry_run: bool) -> None:
    print("\n> " + display_command(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def knn_cache_is_current(directory: Path, shots: list[int]) -> bool:
    required = [directory / f"examples_knn_shot_{shot}.csv" for shot in shots]
    config_path = directory / "retrieval_config.json"
    if not all(path.exists() for path in [*required, config_path]):
        return False
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return bool(config.get("timing_protocol"))


def backend_args(args: argparse.Namespace) -> list[str]:
    values = ["--model", args.model, "--backend", args.backend, "--max-tokens", "256"]
    if args.backend == "transformers":
        values.extend(
            ["--torch-dtype", args.torch_dtype, "--device-map", args.device_map]
        )
    else:
        values.extend(
            [
                "--api-base",
                args.api_base,
                "--api-key",
                args.api_key,
                "--temperature",
                "0",
            ]
        )
    values.extend(["--prompt-mode", "minimal", "--invalid-retries", "0"])
    if args.resume:
        values.append("--resume")
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    return values


def build_examples_command(
    args: argparse.Namespace,
    *,
    manifest: str,
    strategy: str,
    out_dir: Path,
    run_index: int | None = None,
) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "build_examples.py"),
        "--manifest-dir",
        manifest,
        "--strategy",
        strategy,
        "--shots",
        *[str(shot) for shot in args.shots],
        "--out-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
    ]
    if strategy == "random":
        assert run_index is not None
        command.extend(["--run-index", str(run_index)])
    else:
        command.extend(
            [
                "--knn-scope",
                "global",
                "--feature-backend",
                "remoteclip",
                "--remoteclip-cache",
                args.remoteclip_cache,
                "--feature-batch-size",
                str(args.feature_batch_size),
                "--feature-num-workers",
                str(args.feature_num_workers),
            ]
        )
        if args.remoteclip_checkpoint:
            command.extend(["--remoteclip-checkpoint", args.remoteclip_checkpoint])
    return command


def aggregate(args: argparse.Namespace, run_dirs: list[Path], output: Path) -> None:
    run(
        [
            args.python,
            str(PROJECT_ROOT / "aggregate_paper_runs.py"),
            *[str(path) for path in run_dirs],
            "--model",
            args.model,
            "--output",
            str(output),
        ],
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    if args.backend == "transformers" and not args.dry_run and not Path(args.model).exists():
        raise FileNotFoundError(f"Local model not found: {args.model}")
    common_backend = backend_args(args)
    out_root = Path(args.out_root) / args.model_alias

    for dataset in args.datasets:
        config = DATASET_CONFIG[dataset]
        manifest = config["manifest"]
        if not args.dry_run:
            validate_eval100_manifest(PROJECT_ROOT / manifest)

        knn_examples = Path(config["knn_examples"])
        if "knn" in args.methods:
            if args.rebuild_knn or (
                not args.dry_run and not knn_cache_is_current(knn_examples, args.shots)
            ):
                run(
                    build_examples_command(
                        args,
                        manifest=manifest,
                        strategy="knn",
                        out_dir=knn_examples,
                    ),
                    dry_run=args.dry_run,
                )

        zero_dirs: list[Path] = []
        random_dirs = {shot: [] for shot in args.shots}
        knn_dirs = {shot: [] for shot in args.shots}
        for run_index in range(RUNS):
            run_name = f"run_{run_index + 1:02d}"
            if "zero" in args.methods:
                out_dir = out_root / dataset / "zero" / run_name
                zero_dirs.append(out_dir)
                run(
                    [
                        args.python,
                        str(PROJECT_ROOT / "run_zero_shot_mllm.py"),
                        "--manifest-dir",
                        manifest,
                        *common_backend,
                        "--out-dir",
                        str(out_dir),
                    ],
                    dry_run=args.dry_run,
                )

            random_examples = out_root / dataset / "random_examples" / run_name
            if "random" in args.methods:
                run(
                    build_examples_command(
                        args,
                        manifest=manifest,
                        strategy="random",
                        out_dir=random_examples,
                        run_index=run_index,
                    ),
                    dry_run=args.dry_run,
                )

            for shot in args.shots:
                if "random" in args.methods:
                    out_dir = out_root / dataset / f"random_k{shot}" / run_name
                    random_dirs[shot].append(out_dir)
                    run(
                        [
                            args.python,
                            str(PROJECT_ROOT / "run_random_fewshot_mllm.py"),
                            "--manifest-dir",
                            manifest,
                            "--examples-csv",
                            str(random_examples / f"examples_random_shot_{shot}.csv"),
                            *common_backend,
                            "--out-dir",
                            str(out_dir),
                        ],
                        dry_run=args.dry_run,
                    )
                if "knn" in args.methods:
                    out_dir = out_root / dataset / f"knn_k{shot}" / run_name
                    knn_dirs[shot].append(out_dir)
                    run(
                        [
                            args.python,
                            str(PROJECT_ROOT / "run_knn_totalshot_mllm.py"),
                            "--manifest-dir",
                            manifest,
                            "--examples-csv",
                            str(knn_examples / f"examples_knn_shot_{shot}.csv"),
                            *common_backend,
                            "--out-dir",
                            str(out_dir),
                        ],
                        dry_run=args.dry_run,
                    )

        if args.limit is None:
            if zero_dirs:
                aggregate(args, zero_dirs, out_root / dataset / "zero" / "ten_run_summary.json")
            for shot in args.shots:
                if random_dirs[shot]:
                    aggregate(
                        args,
                        random_dirs[shot],
                        out_root / dataset / f"random_k{shot}" / "ten_run_summary.json",
                    )
                if knn_dirs[shot]:
                    aggregate(
                        args,
                        knn_dirs[shot],
                        out_root / dataset / f"knn_k{shot}" / "ten_run_summary.json",
                    )


if __name__ == "__main__":
    main()
