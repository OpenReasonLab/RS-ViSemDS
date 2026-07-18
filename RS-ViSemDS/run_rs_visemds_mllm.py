from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval100_protocol import validate_eval100_manifest
from run_zero_shot_mllm import (
    INVALID_LABEL,
    int_value,
    invalid_retry_instruction,
    parse_prediction,
    summarize_rows,
)
from strict_fewshot.local_mllm import (
    LOCAL_MLLM_IMPLEMENTATION_VERSION,
    TransformersVisionLLM,
)
from strict_fewshot.utils import read_csv, read_json, repo_path, sha256_file, write_csv, write_json

from rs_visemds.category_texts import category_text_sha256
from rs_visemds.prompt_builder import (
    PROMPT_MODES,
    SYSTEM_PROMPT,
    build_local_messages_and_images,
)


PREDICTION_FIELDS = [
    "dataset", "strategy", "shot", "model", "target_id", "target_path",
    "true_label", "pred_label", "correct", "score", "thoughts", "parse_valid",
    "parse_mode", "raw_pred_label", "raw_response", "error", "num_examples",
    "example_labels", "example_paths", "example_ranks", "example_scores",
    "S_img", "S_typ", "S_sem", "selection_seconds", "generation_seconds",
    "predict_seconds", "total_seconds", "attempt_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen open-weight MLLM inference with RS-ViSemDS demonstrations."
    )
    parser.add_argument("--dataset", required=True, choices=["aid", "nwpu_fg_urban"])
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--selected-examples-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--invalid-retries", type=int, default=1)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="legacy")
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=[],
        help="Evaluate only these classes while retaining the full candidate label set.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invocation_start = time.perf_counter()
    manifest_dir = repo_path(args.manifest_dir, PROJECT_ROOT)
    selected_path = repo_path(args.selected_examples_csv, PROJECT_ROOT)
    out_dir = repo_path(args.out_dir, PROJECT_ROOT)
    validate_eval100_manifest(manifest_dir)
    summary = read_json(manifest_dir / "summary.json")
    class_order = read_json(manifest_dir / "class_order.json")["classes"]
    data_root = repo_path(summary["data_root"], PROJECT_ROOT)
    full_evaluation = read_csv(manifest_dir / "evaluation.csv")
    support = read_csv(manifest_dir / "support.csv")
    examples_by_target, k = load_and_validate_examples(
        selected_path, full_evaluation, support
    )
    target_classes = args.target_classes or list(class_order)
    unknown_target_classes = sorted(set(target_classes) - set(class_order))
    if unknown_target_classes:
        raise ValueError(f"Unknown target classes: {unknown_target_classes}")
    target_class_set = set(target_classes)
    evaluation = [
        row for row in full_evaluation if row["label"] in target_class_set
    ]
    if args.limit is not None:
        evaluation = evaluation[:args.limit]
    missing = [row["target_id"] for row in evaluation if row["target_id"] not in examples_by_target]
    if missing:
        raise ValueError(f"Selections missing for evaluation targets: {missing[:5]}")

    model = TransformersVisionLLM(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    selection_config_path = selected_path.parent / "selection_config.json"
    selection_config = read_json(selection_config_path) if selection_config_path.exists() else {}
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "RS-ViSemDS",
        "dataset": summary["dataset"],
        "manifest_dir": str(manifest_dir),
        "selected_examples_csv": str(selected_path),
        "selection_config": selection_config,
        "model": args.model,
        "backend": "transformers",
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "max_tokens": args.max_tokens,
        "do_sample": False,
        "shot": k,
        "shot_definition": "query_specific_total_labeled_images",
        "system_prompt": SYSTEM_PROMPT,
        "prompt_mode": args.prompt_mode,
        "target_classes": target_classes,
        "prompt_builder_sha256": sha256_file(
            Path(__file__).resolve().parent / "rs_visemds" / "prompt_builder.py"
        ),
        "category_text_sha256": category_text_sha256(args.dataset, class_order),
        "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        "support_sha256": sha256_file(manifest_dir / "support.csv"),
        "class_order_sha256": sha256_file(manifest_dir / "class_order.json"),
        "selected_examples_sha256": sha256_file(selected_path),
        "selection_config_sha256": (
            sha256_file(selection_config_path) if selection_config_path.exists() else None
        ),
        "local_mllm_implementation_version": LOCAL_MLLM_IMPLEMENTATION_VERSION,
        "conversation_context": "stateless_new_context_per_target",
        "num_requested_targets": len(evaluation),
        "timing_note": (
            "total_seconds includes cached-embedding selection time plus prompt construction "
            "and generation; one-time RemoteCLIP cache construction is reported separately"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"
    if args.resume and config_path.exists():
        old = read_json(config_path)
        for key in (
            "dataset", "model", "shot", "evaluation_sha256", "support_sha256",
            "class_order_sha256", "selected_examples_sha256",
            "selection_config_sha256", "category_text_sha256",
            "local_mllm_implementation_version",
            "prompt_mode", "target_classes", "prompt_builder_sha256",
        ):
            if old.get(key) != config.get(key):
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"{old.get(key)!r} != {config.get(key)!r}"
                )
    write_json(config_path, config)

    predictions_path = out_dir / "predictions.csv"
    existing = read_csv(predictions_path) if args.resume and predictions_path.exists() else []
    rows = [
        row for row in existing
        if not row.get("error") and int_value(row.get("parse_valid")) == 1
    ]
    completed = {row["target_id"] for row in rows}

    for index, target in enumerate(evaluation, start=1):
        if target["target_id"] in completed:
            continue
        examples = examples_by_target[target["target_id"]]
        selection_seconds = float(examples[0].get("selection_seconds") or 0.0)
        parsed = _empty_prediction()
        raw_text = ""
        error = ""
        generation_seconds = 0.0
        prompt_and_generation_seconds = 0.0
        retry_note = ""
        attempt_count = 0
        for attempt_count in range(1, max(0, args.invalid_retries) + 2):
            try:
                prompt_start = time.perf_counter()
                messages, images = build_local_messages_and_images(
                    data_root=data_root,
                    target_path=target["path"],
                    dataset=args.dataset,
                    class_order=class_order,
                    examples=examples,
                    retry_instruction=retry_note,
                    prompt_mode=args.prompt_mode,
                )
                generation_start = time.perf_counter()
                raw_text = model.generate_from_messages(messages, images)
                generation_seconds += time.perf_counter() - generation_start
                prompt_and_generation_seconds += time.perf_counter() - prompt_start
                parsed = parse_prediction(raw_text, class_order)
                if parsed["parse_valid"] == 1:
                    break
                if attempt_count <= args.invalid_retries:
                    retry_note = invalid_retry_instruction(class_order, raw_text)
                    continue
                break
            except (RuntimeError, ValueError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break

        if error:
            parsed = _empty_prediction(parse_mode="request_error")
        correct = int(
            not error and parsed["parse_valid"] == 1
            and parsed["pred_label"] == target["label"]
        )
        total_seconds = selection_seconds + prompt_and_generation_seconds
        row = {
            "dataset": summary["dataset"],
            "strategy": "rs_visemds",
            "shot": k,
            "model": args.model,
            "target_id": target["target_id"],
            "target_path": target["path"],
            "true_label": target["label"],
            "pred_label": parsed["pred_label"],
            "correct": correct,
            "score": parsed["score"],
            "thoughts": parsed["thoughts"],
            "parse_valid": parsed["parse_valid"],
            "parse_mode": parsed["parse_mode"],
            "raw_pred_label": parsed["raw_pred_label"],
            "raw_response": raw_text,
            "error": error,
            "num_examples": len(examples),
            "example_labels": "|".join(item["example_label"] for item in examples),
            "example_paths": "|".join(item["example_path"] for item in examples),
            "example_ranks": "|".join(item["rank"] for item in examples),
            "example_scores": "|".join(item["score"] for item in examples),
            "S_img": "|".join(item["S_img"] for item in examples),
            "S_typ": "|".join(item["S_typ"] for item in examples),
            "S_sem": "|".join(item["S_sem"] for item in examples),
            "selection_seconds": f"{selection_seconds:.8f}",
            "generation_seconds": f"{generation_seconds:.4f}",
            "predict_seconds": f"{total_seconds:.4f}",
            "total_seconds": f"{total_seconds:.4f}",
            "attempt_count": attempt_count,
        }
        rows.append(row)
        completed.add(target["target_id"])
        print(
            f"{index}/{len(evaluation)} {target['target_id']}: "
            f"true={target['label']} pred={parsed['pred_label']} correct={correct}"
        )
        if len(rows) % args.save_every == 0:
            write_outputs(
                out_dir, rows, class_order, args.bootstrap_samples,
                args.bootstrap_seed, len(evaluation),
                time.perf_counter() - invocation_start,
            )

    write_outputs(
        out_dir, rows, class_order, args.bootstrap_samples, args.bootstrap_seed,
        len(evaluation), time.perf_counter() - invocation_start,
    )
    print(f"Wrote RS-ViSemDS MLLM results to {out_dir}")


def load_and_validate_examples(selected_path, evaluation, support):
    rows = read_csv(selected_path)
    if not rows:
        raise ValueError(f"No rows in {selected_path}")
    strategies = {row["strategy"] for row in rows}
    shots = {int(row["shot"]) for row in rows}
    if strategies != {"rs_visemds"} or len(shots) != 1:
        raise ValueError(f"Expected one RS-ViSemDS shot, got {strategies=} {shots=}")
    k = next(iter(shots))
    eval_by_id = {row["target_id"]: row for row in evaluation}
    support_pairs = {(row["label"], row["path"]) for row in support}
    grouped = defaultdict(list)
    for row in rows:
        target = eval_by_id.get(row["target_id"])
        if target is None:
            raise ValueError(f"Unknown target: {row['target_id']}")
        if row["target_path"] != target["path"] or row["target_label"] != target["label"]:
            raise ValueError(f"Target metadata mismatch: {row['target_id']}")
        if (row["example_label"], row["example_path"]) not in support_pairs:
            raise ValueError(f"Example outside support pool: {row['example_path']}")
        grouped[row["target_id"]].append(row)
    for target_id, items in grouped.items():
        ordered = sorted(items, key=lambda row: int(row["rank"]))
        if len(ordered) != k:
            raise ValueError(f"Target {target_id} has {len(ordered)} examples; expected {k}")
        if [int(row["rank"]) for row in ordered] != list(range(1, k + 1)):
            raise ValueError(f"Non-contiguous ranks for {target_id}")
        if len({row["example_path"] for row in ordered}) != k:
            raise ValueError(f"Duplicate examples for {target_id}")
        scores = [float(row["score"]) for row in ordered]
        if any(scores[i] < scores[i + 1] for i in range(k - 1)):
            raise ValueError(f"Scores are not descending for {target_id}")
        grouped[target_id] = ordered
    return dict(grouped), k


def write_outputs(out_dir, rows, class_order, bootstrap_samples, bootstrap_seed,
                  expected_targets, wall_seconds):
    write_csv(out_dir / "predictions.csv", rows, PREDICTION_FIELDS)
    metrics, per_class_rows, matrices = summarize_rows(
        rows, class_order, bootstrap_samples, bootstrap_seed
    )
    by_model = defaultdict(list)
    for row in per_class_rows:
        by_model[row["model"]].append(row)
    for model_name, model_rows in by_model.items():
        metrics[model_name]["macro_precision"] = sum(
            float(row["precision"]) for row in model_rows
        ) / len(model_rows)
        metrics[model_name]["macro_recall"] = sum(
            float(row["accuracy"]) for row in model_rows
        ) / len(model_rows)
    write_json(out_dir / "summary.json", {
        "metrics": metrics,
        "num_completed_targets": len(rows),
        "num_expected_targets": expected_targets,
        "completion_rate": len(rows) / expected_targets if expected_targets else None,
        "invocation_wall_seconds": wall_seconds,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_csv(
        out_dir / "per_class_accuracy.csv", per_class_rows,
        ["model", "class", "support", "accuracy", "precision", "f1", "invalid_predictions"],
    )
    pred_columns = [*class_order, INVALID_LABEL]
    for model_name, matrix in matrices.items():
        matrix_rows = []
        for true_label, values in zip(class_order, matrix):
            row = {"true_label": true_label}
            row.update(dict(zip(pred_columns, values)))
            matrix_rows.append(row)
        write_csv(
            out_dir / f"confusion_matrix_{Path(model_name).name}.csv",
            matrix_rows,
            ["true_label", *pred_columns],
        )


def _empty_prediction(parse_mode: str = "invalid") -> dict:
    return {
        "pred_label": INVALID_LABEL,
        "raw_pred_label": "",
        "parse_valid": 0,
        "parse_mode": parse_mode,
        "score": "",
        "thoughts": "",
    }


if __name__ == "__main__":
    main()
