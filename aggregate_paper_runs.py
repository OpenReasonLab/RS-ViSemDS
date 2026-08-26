from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the paper's arithmetic mean over ten independent runs."
    )
    parser.add_argument("run_dirs", nargs="+", help="Ten directories containing summary.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="",
        help="Model key used by MLLM summary.json files; unnecessary for RS-ViSemDS.",
    )
    return parser.parse_args()


def load_metrics(run_dir: Path, model: str = "") -> dict[str, float]:
    path = run_dir / "summary.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("metrics", payload)
    if model:
        if model not in values:
            raise ValueError(f"{path} does not contain model key {model!r}")
        values = values[model]
    elif "overall_accuracy" not in values and "accuracy" not in values:
        model_rows = [value for value in values.values() if isinstance(value, dict)]
        if len(model_rows) == 1:
            values = model_rows[0]
    if "accuracy" not in values and "overall_accuracy" in values:
        values = {**values, "accuracy": values["overall_accuracy"]}
    missing = [key for key in METRICS if key not in values]
    if missing:
        raise ValueError(f"{path} is missing metrics: {missing}")
    return {key: float(values[key]) for key in METRICS}


def main() -> None:
    args = parse_args()
    if len(args.run_dirs) != 10:
        raise ValueError(f"The paper protocol requires exactly 10 runs, got {len(args.run_dirs)}")
    run_dirs = [Path(value).resolve() for value in args.run_dirs]
    if len(set(run_dirs)) != 10:
        raise ValueError("run directories must be distinct")
    per_run = [load_metrics(path, args.model) for path in run_dirs]
    output = {
        "aggregation": "arithmetic_mean_over_ten_independent_runs",
        "num_runs": 10,
        "run_dirs": [str(path) for path in run_dirs],
        "model": args.model or None,
        "per_run": per_run,
        "mean": {
            key: float(np.mean([row[key] for row in per_run], dtype=np.float64))
            for key in METRICS
        },
        "population_std": {
            key: float(np.std([row[key] for row in per_run], ddof=0, dtype=np.float64))
            for key in METRICS
        },
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    print(f"Wrote ten-run aggregate to {output_path}")


if __name__ == "__main__":
    main()
