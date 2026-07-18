from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from eval100_protocol import DATASET_CONFIG, validate_eval100_manifest

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GPT-4o zero-shot, random retrieval augmentation, and RemoteCLIP-kNN "
            "retrieval augmentation experiments on AID and NWPU-FG-Urban."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["aid", "nwpu_fg_urban"], choices=sorted(DATASET_CONFIG))
    parser.add_argument("--shots", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--model", default=os.environ.get("GPT4O_MODEL", "gpt-4o"))
    parser.add_argument("--out-root", default="GPT-4o/results_eval100_seed42")
    parser.add_argument("--backend", choices=["api", "transformers"], default="api")
    parser.add_argument(
        "--api-base",
        default=(
            os.environ.get("ZZZ_API_BASE")
            or os.environ.get("MLLM_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.zhizengzeng.com/v1"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("ZZZ_API_KEY")
            or os.environ.get("MLLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ),
    )
    parser.add_argument("--prompt-mode", choices=["minimal", "guided"], default="minimal")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-zero", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def quote_arg(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return repr(value)
    return value


def command_for_display(command: list[str]) -> list[str]:
    display = list(command)
    for index, part in enumerate(display[:-1]):
        if part == "--api-key":
            display[index + 1] = "***"
    return display


def run_command(command: list[str], dry_run: bool) -> None:
    display_command = command_for_display(command)
    print("\n> " + " ".join(quote_arg(part) for part in display_command), flush=True)
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        safe_command = " ".join(quote_arg(part) for part in display_command)
        raise SystemExit(
            f"Command failed with exit code {exc.returncode}: {safe_command}"
        ) from None


def backend_args(args: argparse.Namespace) -> list[str]:
    common = ["--model", args.model, "--backend", args.backend]
    if args.backend == "api":
        common.extend([
            "--api-base", args.api_base,
            "--api-key", args.api_key,
            "--temperature", str(args.temperature),
            "--timeout", str(args.timeout),
            "--retries", str(args.retries),
        ])
    else:
        common.extend([
            "--torch-dtype", args.torch_dtype,
            "--device-map", args.device_map,
        ])
    common.extend([
        "--prompt-mode", args.prompt_mode,
        "--max-tokens", str(args.max_tokens),
        "--resume",
    ])
    if args.limit is not None:
        common.extend(["--limit", str(args.limit)])
    return common


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    if args.backend == "api" and not args.api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set ZZZ_API_KEY/MLLM_API_KEY/OPENAI_API_KEY, or pass --api-key.")
    if args.backend == "transformers" and not args.dry_run:
        require_dir(Path(args.model))

    zero_script = root / "run_zero_shot_mllm.py"
    random_script = root / "run_random_fewshot_mllm.py"
    knn_script = root / "run_knn_totalshot_mllm.py"
    if not args.skip_zero:
        require_file(zero_script)
    if not args.skip_random:
        require_file(random_script)
    if not args.skip_knn:
        require_file(knn_script)

    out_root = root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    base_backend_args = backend_args(args)

    for dataset in args.datasets:
        cfg = DATASET_CONFIG[dataset]
        manifest_dir = root / cfg["manifest"]
        random_dir = root / cfg["random_examples"]
        knn_dir = root / cfg["knn_examples"]
        require_dir(manifest_dir)
        require_file(manifest_dir / "evaluation.csv")
        require_file(manifest_dir / "support.csv")
        require_file(manifest_dir / "class_order.json")
        require_file(manifest_dir / "summary.json")
        validate_eval100_manifest(manifest_dir)

        if not args.skip_zero:
            command = [
                args.python,
                str(zero_script),
                "--manifest-dir", cfg["manifest"],
                *base_backend_args,
                "--out-dir", f"{args.out_root}/gpt4o_zero_{dataset}",
            ]
            run_command(command, args.dry_run)

        for shot in args.shots:
            if shot <= 0:
                raise ValueError(f"Shot must be positive, got {shot}")

            if not args.skip_random:
                random_csv = random_dir / f"examples_random_shot_{shot}.csv"
                require_file(random_csv)
                command = [
                    args.python,
                    str(random_script),
                    "--manifest-dir", cfg["manifest"],
                    "--examples-csv", f"{cfg['random_examples']}/examples_random_shot_{shot}.csv",
                    *base_backend_args,
                    "--out-dir", f"{args.out_root}/gpt4o_random_{dataset}_shot{shot}",
                ]
                run_command(command, args.dry_run)

            if not args.skip_knn:
                knn_csv = knn_dir / f"examples_knn_shot_{shot}.csv"
                require_file(knn_csv)
                command = [
                    args.python,
                    str(knn_script),
                    "--manifest-dir", cfg["manifest"],
                    "--examples-csv", f"{cfg['knn_examples']}/examples_knn_shot_{shot}.csv",
                    *base_backend_args,
                    "--out-dir", f"{args.out_root}/gpt4o_knn_{dataset}_shot{shot}",
                ]
                run_command(command, args.dry_run)

    print("\nGPT-4o suite finished.", flush=True)


if __name__ == "__main__":
    main()
