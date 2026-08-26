from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eval100_protocol import validate_eval100_manifest
from run_zero_shot_mllm import INVALID_LABEL, parse_prediction
from strict_fewshot.local_mllm import (
    LOCAL_MLLM_IMPLEMENTATION_VERSION,
    TransformersVisionLLM,
)
from strict_fewshot.utils import read_json, repo_path, sha256_file

from rs_visemds.adaptive_weights import (
    BASE_WEIGHTS,
    compute_adaptive_weights,
)
from rs_visemds.calibration import calibrate_temperatures
from rs_visemds.category_texts import (
    category_descriptions_sha256,
    category_text_sha256,
)
from rs_visemds.embedding_backend import load_or_build_embeddings
from rs_visemds.prompt_builder import (
    SYSTEM_PROMPT,
    build_local_messages_and_images,
    category_rules_sha256,
    prompt_template_sha256,
)
from rs_visemds.selector import (
    class_balanced_candidates,
    select_adaptive_demonstrations,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
WEIGHT_MODE = "adaptive"
PROMPT_MODE = "paper_v1"
DATASETS = {
    "aid": {
        "embedding_name": "aid",
        "manifest_dir": "manifests/aid_eval100_seed42",
        "display_name": "AID",
    },
    "nwpu_urban": {
        "embedding_name": "nwpu_fg_urban",
        "manifest_dir": "manifests/nwpu_eval100_seed42",
        "display_name": "NWPU-Urban",
    },
}
RESUME_KEYS = (
    "method",
    "dataset_argument",
    "model_id",
    "weight_mode",
    "base_weights",
    "visual_temperature",
    "semantic_temperature",
    "calibration_seed",
    "calibration_samples_per_class",
    "candidates_per_class",
    "num_demonstrations",
    "prompt_mode",
    "torch_dtype",
    "device_map",
    "max_tokens",
    "min_pixels",
    "max_pixels",
    "diagnostic_only",
    "target_ids_sha256",
    "evaluation_sha256",
    "support_sha256",
    "class_order_sha256",
    "category_text_sha256",
    "category_descriptions_sha256",
    "category_rules_sha256",
    "prompt_template_sha256",
    "runner_sha256",
    "adaptive_module_sha256",
    "selector_sha256",
    "prompt_builder_sha256",
    "category_texts_module_sha256",
    "embedding_backend_sha256",
    "local_mllm_implementation_version",
    "limit",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the target-adaptive RS-ViSemDS experiment."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--weight-mode", required=True, choices=[WEIGHT_MODE])
    parser.add_argument("--candidates-per-class", type=int, default=3)
    parser.add_argument("--num-demonstrations", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-dir", default="")
    parser.add_argument("--cache-dir", default="RS-ViSemDS/cache")
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--calibration-samples-per-class", type=int, default=50)
    parser.add_argument("--temperature-min", type=float, default=1e-3)
    parser.add_argument("--temperature-max", type=float, default=1.0)
    parser.add_argument("--calibration-iterations", type=int, default=48)
    parser.add_argument(
        "--max-run-error-retries",
        type=int,
        default=3,
        help="Maximum resume retries after the first failed execution attempt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test only; omit for both formal experiments.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Mark a target-ID subset run as diagnostic and ineligible for paper metrics.",
    )
    parser.add_argument(
        "--target-ids-file",
        default="",
        help="Diagnostic-only text file containing one target image_id per line.",
    )
    args = parser.parse_args(argv)
    validate_experiment_contract(args)
    return args


def validate_experiment_contract(args: argparse.Namespace) -> None:
    if args.weight_mode != WEIGHT_MODE:
        raise ValueError("Only --weight-mode adaptive is implemented in this experiment")
    if args.candidates_per_class != 3:
        raise ValueError("candidates-per-class is fixed at 3")
    if args.num_demonstrations != 3:
        raise ValueError("num-demonstrations is fixed at 3")
    if not args.model_id.strip():
        raise ValueError("--model-id must not be empty")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if args.calibration_samples_per_class <= 0:
        raise ValueError("--calibration-samples-per-class must be positive")
    if args.calibration_iterations <= 0:
        raise ValueError("--calibration-iterations must be positive")
    if not 0.0 < args.temperature_min < args.temperature_max:
        raise ValueError("temperature bounds must satisfy 0 < min < max")
    if args.max_run_error_retries < 0:
        raise ValueError("--max-run-error-retries must be non-negative")
    if args.target_ids_file and not args.diagnostic_only:
        raise ValueError(
            "--target-ids-file requires --diagnostic-only"
        )
    if args.diagnostic_only and not (args.target_ids_file or args.limit is not None):
        raise ValueError("--diagnostic-only requires --target-ids-file or --limit")
    if args.target_ids_file and args.limit is not None:
        raise ValueError("--target-ids-file cannot be combined with --limit")


def main() -> None:
    args = parse_args()
    invocation_start = time.perf_counter()
    dataset_spec = DATASETS[args.dataset]
    manifest_arg = args.manifest_dir or dataset_spec["manifest_dir"]
    manifest_dir = repo_path(manifest_arg, PROJECT_ROOT)
    out_dir = repo_path(args.output_dir, PROJECT_ROOT)
    cache_dir = repo_path(args.cache_dir, PROJECT_ROOT)
    remoteclip_cache = repo_path(args.remoteclip_cache, PROJECT_ROOT)
    checkpoint = _optional_path(args.remoteclip_checkpoint)

    validate_eval100_manifest(manifest_dir)
    embedding_start = time.perf_counter()
    bundle = load_or_build_embeddings(
        dataset=dataset_spec["embedding_name"],
        manifest_dir=manifest_dir,
        project_root=PROJECT_ROOT,
        cache_dir=cache_dir,
        remoteclip_cache=remoteclip_cache,
        remoteclip_checkpoint=checkpoint,
        batch_size=args.feature_batch_size,
        num_workers=args.feature_num_workers,
        force=args.force_cache,
    )
    embedding_preparation_seconds = time.perf_counter() - embedding_start
    assert_no_target_leakage(bundle.target_rows, bundle.support_rows)

    evaluation = list(bundle.target_rows)
    if args.target_ids_file:
        target_ids_path = repo_path(args.target_ids_file, PROJECT_ROOT)
        evaluation = filter_evaluation_by_target_ids(evaluation, target_ids_path)
    if args.limit is not None:
        evaluation = evaluation[: args.limit]
    target_index = {
        row["target_id"]: index for index, row in enumerate(bundle.target_rows)
    }
    support_labels = [row["label"] for row in bundle.support_rows]
    calibration_start = time.perf_counter()
    calibration = calibrate_temperatures(
        bundle.support_embeddings,
        support_labels,
        bundle.category_prototypes,
        bundle.class_order,
        samples_per_class=args.calibration_samples_per_class,
        seed=args.calibration_seed,
        minimum_temperature=args.temperature_min,
        maximum_temperature=args.temperature_max,
        iterations=args.calibration_iterations,
    )
    calibration_seconds = time.perf_counter() - calibration_start
    config = build_run_config(
        args, manifest_dir, bundle, len(evaluation), calibration, calibration_seconds
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"
    if args.resume and config_path.exists():
        old_config = read_json(config_path)
        validate_resume_config(old_config, config)
        config["created_at_utc"] = old_config.get(
            "created_at_utc", config["created_at_utc"]
        )
    _write_json_atomic(config_path, config)

    predictions_path = out_dir / "predictions.jsonl"
    records = load_prediction_records(predictions_path) if args.resume else []
    run_errors_path = out_dir / "run_errors.jsonl"
    run_errors = load_prediction_records(run_errors_path) if args.resume else []
    records, migrated_errors = migrate_legacy_error_rows(records)
    run_errors.extend(migrated_errors)
    _validate_resume_records(records, evaluation)
    completed = {record["image_id"] for record in records}

    pending = [
        row
        for row in evaluation
        if row["target_id"] not in completed
        and can_attempt_target(
            row["target_id"], run_errors, args.max_run_error_retries
        )
    ]
    model = None
    if pending:
        try:
            model = TransformersVisionLLM(
                args.model_id,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                max_new_tokens=args.max_tokens,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        except Exception as exc:  # noqa: BLE001 - failure must be persisted for resume
            error_text = f"{type(exc).__name__}: {exc}"
            run_errors.append(
                {
                    "image_id": "__model_initialization__",
                    "attempt": 1
                    + sum(
                        row.get("image_id") == "__model_initialization__"
                        for row in run_errors
                    ),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "error_stage": "model_initialization",
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mllm_call_count": 0,
                    "total_time_seconds": time.perf_counter() - invocation_start,
                }
            )
            runtime_environment = collect_environment_metadata(args.model_id, None)
            runtime_environment["model_initialization_error"] = error_text
            config["runtime_environment"] = runtime_environment
            _write_json_atomic(config_path, config)
            write_environment_file(out_dir / "environment.txt", runtime_environment)
            _write_all_outputs(
                out_dir=out_dir,
                records=records,
                class_order=bundle.class_order,
                expected_targets=len(evaluation),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                embedding_preparation_seconds=embedding_preparation_seconds,
                invocation_wall_seconds=time.perf_counter() - invocation_start,
                dataset_name=dataset_spec["display_name"],
                model_id=args.model_id,
                run_errors=run_errors,
                expected_target_ids=[row["target_id"] for row in evaluation],
                global_run_error_count=1,
                formal_result_eligible=not args.diagnostic_only,
            )
            print(f"Model initialization failed; incomplete outputs written to {out_dir}")
            return
    runtime_environment = collect_environment_metadata(args.model_id, model)
    config["runtime_environment"] = runtime_environment
    _write_json_atomic(config_path, config)
    write_environment_file(out_dir / "environment.txt", runtime_environment)

    for number, target in enumerate(evaluation, start=1):
        image_id = target["target_id"]
        if image_id in completed:
            continue
        if not can_attempt_target(
            image_id, run_errors, args.max_run_error_retries
        ):
            print(
                f"{number}/{len(evaluation)} {image_id}: run-error retry limit reached"
            )
            continue
        target_embedding = bundle.target_embeddings[target_index[image_id]]
        attempt = next_run_error_attempt(image_id, run_errors)
        prediction_record = process_unlabeled_target(
            image_id=image_id,
            target_path=target["path"],
            target_embedding=target_embedding,
            dataset=dataset_spec["embedding_name"],
            data_root=bundle.data_root,
            class_order=bundle.class_order,
            support_rows=bundle.support_rows,
            support_embeddings=bundle.support_embeddings,
            support_labels=support_labels,
            category_prototypes=bundle.category_prototypes,
            model=model,
            visual_temperature=calibration.visual_temperature,
            semantic_temperature=calibration.semantic_temperature,
            candidates_per_class=args.candidates_per_class,
            num_demonstrations=args.num_demonstrations,
        )
        if prediction_record["error"]:
            run_errors.append(build_run_error_event(prediction_record, attempt))
            _write_checkpoint_outputs(out_dir, records, run_errors)
            print(
                f"{number}/{len(evaluation)} {image_id}: "
                f"run_error attempt={attempt} stage={prediction_record['error_stage']} "
                f"error={prediction_record['error']}"
            )
            continue
        # The ground-truth label is first introduced after retrieval, gating,
        # demonstration selection, prompt construction, and model inference.
        prediction_record["ground_truth"] = target["label"]
        prediction_record["is_correct"] = bool(
            not prediction_record["invalid_output"]
            and not prediction_record["error"]
            and prediction_record["prediction"] == target["label"]
        )
        records.append(prediction_record)
        completed.add(image_id)
        _write_checkpoint_outputs(out_dir, records, run_errors)
        print(
            f"{number}/{len(evaluation)} {image_id}: "
            f"true={target['label']} pred={prediction_record['prediction']} "
            f"correct={int(prediction_record['is_correct'])} "
            f"invalid={int(prediction_record['invalid_output'])} "
            f"error={bool(prediction_record['error'])} "
            "adaptive="
            + json.dumps(
                {
                    key: prediction_record[key]
                    for key in (
                        "visual_concentration",
                        "semantic_concentration",
                        "visual_semantic_disagreement",
                        "adjustment",
                        "alpha",
                        "beta",
                        "gamma",
                    )
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    _write_all_outputs(
        out_dir=out_dir,
        records=records,
        class_order=bundle.class_order,
        expected_targets=len(evaluation),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        embedding_preparation_seconds=embedding_preparation_seconds,
        invocation_wall_seconds=time.perf_counter() - invocation_start,
        dataset_name=dataset_spec["display_name"],
        model_id=args.model_id,
        run_errors=run_errors,
        expected_target_ids=[row["target_id"] for row in evaluation],
        formal_result_eligible=not args.diagnostic_only,
    )
    print(f"Wrote Adaptive RS-ViSemDS results to {out_dir}")


def process_unlabeled_target(
    *,
    image_id: str,
    target_path: str,
    target_embedding: np.ndarray,
    dataset: str,
    data_root: Path,
    class_order: list[str],
    support_rows: list[dict[str, str]],
    support_embeddings: np.ndarray,
    support_labels: list[str],
    category_prototypes: np.ndarray,
    model: Any,
    visual_temperature: float,
    semantic_temperature: float,
    candidates_per_class: int,
    num_demonstrations: int,
) -> dict[str, Any]:
    """Process one target without accepting or observing its ground-truth label."""
    target_start = time.perf_counter()
    record = empty_prediction_record(image_id, target_path)
    try:
        retrieval_start = time.perf_counter()
        candidate_indices = class_balanced_candidates(
            target_embedding,
            support_embeddings,
            support_labels,
            class_order,
            candidates_per_class,
        )
        record["retrieval_time_seconds"] = time.perf_counter() - retrieval_start

        selection_start = time.perf_counter()
        adaptive = compute_adaptive_weights(
            target_embedding,
            support_embeddings,
            support_labels,
            category_prototypes,
            class_order,
            visual_temperature=visual_temperature,
            semantic_temperature=semantic_temperature,
        )
        selected, ranked = select_adaptive_demonstrations(
            target_embedding=target_embedding,
            support_embeddings=support_embeddings,
            support_labels=support_labels,
            category_prototypes=category_prototypes,
            class_order=class_order,
            candidate_indices=candidate_indices,
            weights=adaptive.weights,
            k=num_demonstrations,
        )
        record.update(
            {
                "visual_temperature": adaptive.visual_temperature,
                "semantic_temperature": adaptive.semantic_temperature,
                "visual_concentration": adaptive.visual_concentration,
                "semantic_concentration": adaptive.semantic_concentration,
                "visual_semantic_disagreement": adaptive.visual_semantic_disagreement,
                "adjustment": adaptive.adjustment,
                "base_visual_proportion": adaptive.base_visual_proportion,
                "visual_proportion": adaptive.visual_proportion,
                "alpha": adaptive.alpha,
                "beta": adaptive.beta,
                "gamma": adaptive.gamma,
                "visual_class_evidence": list(adaptive.visual_class_evidence),
                "semantic_class_evidence": list(adaptive.semantic_class_evidence),
                "visual_class_distribution": list(
                    adaptive.visual_class_distribution
                ),
                "semantic_class_distribution": list(
                    adaptive.semantic_class_distribution
                ),
            }
        )
        examples = []
        selected_audit = []
        for rank, candidate in enumerate(selected, start=1):
            support = support_rows[candidate.support_index]
            examples.append(
                {
                    "example_label": support["label"],
                    "example_path": support["path"],
                }
            )
            selected_audit.append(
                {
                    "rank": rank,
                    "support_index": candidate.support_index,
                    "label": support["label"],
                    "path": support["path"],
                    "score": candidate.score,
                    "S_img": candidate.s_img,
                    "S_typ": candidate.s_typ,
                    "S_sem": candidate.s_sem,
                    "S_img_norm": candidate.s_img_norm,
                    "S_typ_norm": candidate.s_typ_norm,
                    "S_sem_norm": candidate.s_sem_norm,
                }
            )
        record["selected_demonstrations"] = selected_audit
        record["example_selection_time_seconds"] = (
            time.perf_counter() - selection_start
        )

        prompt_start = time.perf_counter()
        messages, images = build_local_messages_and_images(
            data_root=data_root,
            target_path=target_path,
            dataset=dataset,
            class_order=class_order,
            examples=examples,
        )
        record["prompt_build_time_seconds"] = time.perf_counter() - prompt_start

        inference_start = time.perf_counter()
        record["mllm_call_count"] = 1
        try:
            raw_output = model.generate_from_messages(messages, images)
        finally:
            record["qwen_inference_time_seconds"] = (
                time.perf_counter() - inference_start
            )
        record["raw_model_output"] = raw_output
        parsed = parse_prediction(raw_output, class_order)
        record["prediction"] = parsed["pred_label"]
        record["parse_valid"] = bool(parsed["parse_valid"])
        record["parse_mode"] = parsed["parse_mode"]
        record["raw_pred_label"] = parsed["raw_pred_label"]
        record["model_score"] = parsed["score"]
        record["model_thoughts"] = parsed["thoughts"]
        record["invalid_output"] = not record["parse_valid"]
    except (RuntimeError, ValueError, OSError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["exception_type"] = type(exc).__name__
        record["exception_message"] = str(exc)
        record["traceback"] = traceback.format_exc()
        record["error_stage"] = _infer_error_stage(record)
        record["prediction"] = INVALID_LABEL
        record["parse_valid"] = False
        record["parse_mode"] = "run_error"
    finally:
        record["total_time_seconds"] = time.perf_counter() - target_start
    return record


def empty_prediction_record(image_id: str, target_path: str) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "target_path": target_path,
        "ground_truth": None,
        "prediction": INVALID_LABEL,
        "is_correct": False,
        "visual_temperature": None,
        "semantic_temperature": None,
        "visual_concentration": None,
        "semantic_concentration": None,
        "visual_semantic_disagreement": None,
        "adjustment": None,
        "base_visual_proportion": None,
        "visual_proportion": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
        "visual_class_evidence": [],
        "semantic_class_evidence": [],
        "visual_class_distribution": [],
        "semantic_class_distribution": [],
        "selected_demonstrations": [],
        "raw_model_output": "",
        "parse_valid": False,
        "parse_mode": "",
        "raw_pred_label": "",
        "model_score": "",
        "model_thoughts": "",
        "invalid_output": False,
        "error": "",
        "error_stage": "",
        "exception_type": "",
        "exception_message": "",
        "traceback": "",
        "mllm_call_count": 0,
        "retrieval_time_seconds": 0.0,
        "example_selection_time_seconds": 0.0,
        "prompt_build_time_seconds": 0.0,
        "qwen_inference_time_seconds": 0.0,
        "total_time_seconds": 0.0,
    }


def build_run_config(
    args,
    manifest_dir: Path,
    bundle,
    requested_targets: int,
    calibration,
    calibration_seconds: float,
) -> dict:
    target_ids_path = (
        repo_path(args.target_ids_file, PROJECT_ROOT) if args.target_ids_file else None
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Adaptive RS-ViSemDS",
        "dataset_argument": args.dataset,
        "dataset": DATASETS[args.dataset]["display_name"],
        "manifest_dir": str(manifest_dir),
        "model_id": args.model_id,
        "weight_mode": WEIGHT_MODE,
        "base_weights": dict(zip(("alpha", "beta", "gamma"), BASE_WEIGHTS)),
        "visual_temperature": calibration.visual_temperature,
        "semantic_temperature": calibration.semantic_temperature,
        "temperature_calibration": {
            **calibration.to_dict(),
            "elapsed_seconds": calibration_seconds,
            "objective": "separate multiclass NLL on support-only stratified queries",
            "query_exclusion": (
                "each held-out support query is excluded from its class visual references"
            ),
            "temperature_search_bounds": [args.temperature_min, args.temperature_max],
            "iterations": args.calibration_iterations,
        },
        "calibration_seed": args.calibration_seed,
        "calibration_samples_per_class": args.calibration_samples_per_class,
        "candidates_per_class": args.candidates_per_class,
        "num_demonstrations": args.num_demonstrations,
        "candidate_pool_size": args.candidates_per_class * len(bundle.class_order),
        "score_formula": (
            "R_i=alpha(x)*S_img_norm+beta(x)*S_typ_norm+gamma(x)*S_sem_norm"
        ),
        "adaptive_weight_formula": {
            "concentration": "kappa_b=1-H(p_b)/log(C)",
            "disagreement": "d_vs=JSD(p_vis,p_sem)/log(2)",
            "adjustment": "g=d_vs*(kappa_vis-kappa_sem)",
            "base_visual_proportion": "pi0=alpha0/(alpha0+gamma0)",
            "visual_proportion": "pi_vis=sigmoid(logit(pi0)+g)",
            "alpha": "(1-beta0)*pi_vis",
            "beta": "beta0",
            "gamma": "(1-beta0)*(1-pi_vis)",
        },
        "visual_class_evidence": (
            "stable log-mean-exp of temperature-scaled target-support cosine over "
            "the complete class support set"
        ),
        "semantic_class_evidence": "target-category-prototype cosine / T_sem",
        "class_evidence_distribution": "softmax of class log-evidence",
        "score_normalization": "component-wise min-max over the r*C candidate pool",
        "selection_order": (
            "pure global Top-k by descending R; ties by descending S_img then support index"
        ),
        "prompt_mode": PROMPT_MODE,
        "system_prompt": SYSTEM_PROMPT,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "max_tokens": args.max_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "diagnostic_only": args.diagnostic_only,
        "formal_result_eligible": not args.diagnostic_only,
        "target_ids_file": str(target_ids_path) if target_ids_path else "",
        "target_ids_sha256": sha256_file(target_ids_path) if target_ids_path else None,
        "do_sample": False,
        "maximum_mllm_calls_per_target": 1,
        "invalid_output_retry": False,
        "max_run_error_retries": args.max_run_error_retries,
        "resume_policy": (
            "successful and invalid-output predictions are complete; run-error targets "
            "remain pending and are retried by --resume up to the configured limit"
        ),
        "class_order": bundle.class_order,
        "num_requested_targets": requested_targets,
        "limit": args.limit,
        "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        "support_sha256": sha256_file(manifest_dir / "support.csv"),
        "class_order_sha256": sha256_file(manifest_dir / "class_order.json"),
        "category_text_sha256": category_text_sha256(
            DATASETS[args.dataset]["embedding_name"], bundle.class_order
        ),
        "category_descriptions_sha256": category_descriptions_sha256(
            DATASETS[args.dataset]["embedding_name"], bundle.class_order
        ),
        "category_rules_sha256": category_rules_sha256(
            DATASETS[args.dataset]["embedding_name"],
            bundle.class_order,
        ),
        "prompt_template_sha256": prompt_template_sha256(
            DATASETS[args.dataset]["embedding_name"],
            bundle.class_order,
            args.num_demonstrations,
        ),
        "runner_sha256": sha256_file(PACKAGE_ROOT / "run_rs_visemds.py"),
        "adaptive_module_sha256": sha256_file(
            PACKAGE_ROOT / "rs_visemds" / "adaptive_weights.py"
        ),
        "selector_sha256": sha256_file(PACKAGE_ROOT / "rs_visemds" / "selector.py"),
        "prompt_builder_sha256": sha256_file(
            PACKAGE_ROOT / "rs_visemds" / "prompt_builder.py"
        ),
        "category_texts_module_sha256": sha256_file(
            PACKAGE_ROOT / "rs_visemds" / "category_texts.py"
        ),
        "embedding_backend_sha256": sha256_file(
            PACKAGE_ROOT / "rs_visemds" / "embedding_backend.py"
        ),
        "embedding_metadata": bundle.metadata,
        "local_mllm_implementation_version": LOCAL_MLLM_IMPLEMENTATION_VERSION,
        "label_isolation": (
            "ground truth enters only after process_unlabeled_target returns"
        ),
        "timing_note": (
            "per-image total excludes one-time embedding preparation; retrieval, "
            "selection, prompt construction, and Qwen inference are recorded separately"
        ),
    }


def validate_resume_config(old: dict, new: dict) -> None:
    for key in RESUME_KEYS:
        if old.get(key) != new.get(key):
            raise ValueError(
                f"Resume configuration mismatch for {key}: "
                f"{old.get(key)!r} != {new.get(key)!r}"
            )


def filter_evaluation_by_target_ids(
    evaluation: list[dict[str, str]], target_ids_path: Path
) -> list[dict[str, str]]:
    requested = [
        value.strip()
        for value in target_ids_path.read_text(encoding="utf-8-sig").splitlines()
        if value.strip()
    ]
    if not requested:
        raise ValueError("target IDs file must contain at least one image_id")
    if len(requested) != len(set(requested)):
        raise ValueError("target IDs file contains duplicate image_id values")
    by_id = {row["target_id"]: row for row in evaluation}
    missing = [image_id for image_id in requested if image_id not in by_id]
    if missing:
        raise ValueError(f"Unknown diagnostic target IDs: {missing[:5]}")
    return [by_id[image_id] for image_id in requested]


def assert_no_target_leakage(
    evaluation: list[dict[str, str]], support: list[dict[str, str]]
) -> None:
    evaluation_paths = {row["path"] for row in evaluation}
    support_paths = {row["path"] for row in support}
    overlap = evaluation_paths & support_paths
    if overlap:
        raise ValueError(f"Evaluation/support image leakage: {sorted(overlap)[:5]}")
    evaluation_ids = {row["target_id"] for row in evaluation}
    support_ids = {row.get("support_id", "") for row in support}
    id_overlap = (evaluation_ids & support_ids) - {""}
    if id_overlap:
        raise ValueError(f"Evaluation/support ID leakage: {sorted(id_overlap)[:5]}")


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid resume JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return records


def migrate_legacy_error_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Move pre-policy run-error rows out of predictions on first resume."""
    completed = []
    migrated_errors = []
    for record in records:
        if record.get("error"):
            migrated_errors.append(build_run_error_event(record, attempt=1))
        else:
            completed.append(record)
    return completed, migrated_errors


def next_run_error_attempt(image_id: str, run_errors: list[dict]) -> int:
    attempts = [
        int(row.get("attempt") or 0)
        for row in run_errors
        if row.get("image_id") == image_id
    ]
    return max(attempts, default=0) + 1


def can_attempt_target(
    image_id: str,
    run_errors: list[dict],
    max_run_error_retries: int,
) -> bool:
    # One initial attempt plus the configured number of resume retries.
    return next_run_error_attempt(image_id, run_errors) <= 1 + max_run_error_retries


def build_run_error_event(record: dict, attempt: int) -> dict[str, Any]:
    return {
        "image_id": record.get("image_id", ""),
        "target_path": record.get("target_path", ""),
        "attempt": int(attempt),
        "exception_type": record.get("exception_type")
        or str(record.get("error", "")).partition(":")[0],
        "exception_message": record.get("exception_message")
        or str(record.get("error", "")).partition(":")[2].strip(),
        "error_stage": record.get("error_stage", ""),
        "traceback": record.get("traceback", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mllm_call_count": int(record.get("mllm_call_count") or 0),
        "total_time_seconds": float(record.get("total_time_seconds") or 0.0),
    }


def unresolved_run_error_ids(
    records: list[dict],
    run_errors: list[dict],
    expected_target_ids: list[str],
) -> set[str]:
    completed = {row["image_id"] for row in records}
    expected = set(expected_target_ids)
    historical_error_ids = {
        row.get("image_id") for row in run_errors if row.get("image_id") in expected
    }
    return (historical_error_ids - completed) & expected


def _validate_resume_records(records: list[dict], evaluation: list[dict]) -> None:
    expected = {row["target_id"] for row in evaluation}
    ids = [record.get("image_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Resume predictions contain duplicate image_id values")
    unknown = sorted(set(ids) - expected)
    if unknown:
        raise ValueError(f"Resume predictions contain unknown targets: {unknown[:5]}")


def compute_summary(
    records: list[dict[str, Any]],
    class_order: list[str],
    expected_targets: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    embedding_preparation_seconds: float,
    invocation_wall_seconds: float,
    unresolved_run_error_count: int = 0,
    historical_run_error_count: int = 0,
    formal_result_eligible: bool = True,
) -> tuple[dict, list[dict], list[dict]]:
    class_index = {label: index for index, label in enumerate(class_order)}
    matrix = [[0 for _ in range(len(class_order) + 1)] for _ in class_order]
    for record in records:
        true_index = class_index[record["ground_truth"]]
        pred_index = class_index.get(record["prediction"], len(class_order))
        matrix[true_index][pred_index] += 1

    per_class = []
    precisions, recalls, f1_values = [], [], []
    for label, index in class_index.items():
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        per_class.append(
            {
                "class": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "invalid_outputs": matrix[index][-1],
            }
        )

    total = len(records)
    invalid_count = sum(bool(row["invalid_output"]) for row in records)
    run_complete = total == expected_targets and unresolved_run_error_count == 0
    if run_complete:
        result_status = (
            "FORMAL_COMPLETE"
            if formal_result_eligible
            else "DIAGNOSTIC_COMPLETE_NOT_FOR_PAPER"
        )
    else:
        result_status = "INCOMPLETE_NOT_FOR_PAPER"
    metrics = {
        "run_complete": run_complete,
        "result_status": result_status,
        "formal_result_eligible": formal_result_eligible,
        "processed_count": total,
        "num_completed_targets": total,
        "num_expected_targets": expected_targets,
        "completion_rate": total / expected_targets if expected_targets else 0.0,
        "num_correct": sum(bool(row["is_correct"]) for row in records),
        "accuracy": _safe_mean([float(bool(row["is_correct"])) for row in records]),
        "macro_precision": _safe_mean(precisions),
        "macro_recall": _safe_mean(recalls),
        "macro_f1": _safe_mean(f1_values),
        "invalid_output_count": invalid_count,
        "invalid_output_rate": invalid_count / total if total else 0.0,
        "run_error_count": unresolved_run_error_count,
        "run_error_rate": (
            unresolved_run_error_count / expected_targets if expected_targets else 0.0
        ),
        "historical_run_error_attempt_count": historical_run_error_count,
        "average_retrieval_time_seconds": _mean_field(
            records, "retrieval_time_seconds"
        ),
        "average_example_selection_time_seconds": _mean_field(
            records, "example_selection_time_seconds"
        ),
        "average_qwen_inference_time_seconds": _mean_field(
            records, "qwen_inference_time_seconds"
        ),
        "average_total_time_seconds": _mean_field(records, "total_time_seconds"),
        "average_total_time_including_amortized_embedding_seconds": (
            _mean_field(records, "total_time_seconds")
            + (embedding_preparation_seconds / expected_targets if expected_targets else 0.0)
        ),
        "total_mllm_calls": sum(int(row["mllm_call_count"]) for row in records),
        "embedding_preparation_seconds_this_invocation": embedding_preparation_seconds,
        "invocation_wall_seconds": invocation_wall_seconds,
        "accuracy_bootstrap_95_ci": bootstrap_accuracy_ci(
            records, bootstrap_samples, bootstrap_seed
        ),
        "bootstrap_samples": bootstrap_samples,
    }
    confusion_rows = []
    columns = [*class_order, INVALID_LABEL]
    for label, values in zip(class_order, matrix):
        row = {"ground_truth": label}
        row.update(dict(zip(columns, values)))
        confusion_rows.append(row)
    return metrics, per_class, confusion_rows


def adaptive_weight_statistics(records: list[dict]) -> dict:
    valid = [
        row
        for row in records
        if all(_is_finite_number(row.get(key)) for key in ("alpha", "beta", "gamma"))
    ]
    output = {
        "num_images_with_weights": len(valid),
        "standard_deviation_type": "population",
        "beta_is_fixed": True,
        "beta0": BASE_WEIGHTS[1],
    }
    for key in ("alpha", "beta", "gamma"):
        values = np.asarray([float(row[key]) for row in valid], dtype=np.float64)
        output[key] = _distribution_statistics(values)
    for output_key, record_key in (
        ("mean_visual_concentration", "visual_concentration"),
        ("mean_semantic_concentration", "semantic_concentration"),
        ("mean_visual_semantic_disagreement", "visual_semantic_disagreement"),
        ("mean_adjustment", "adjustment"),
    ):
        output[output_key] = _safe_mean(
            [float(row[record_key]) for row in valid if _is_finite_number(row.get(record_key))]
        )
    return output


def class_distribution_statistics(
    records: list[dict], class_order: list[str]
) -> dict[str, Any]:
    usable = [
        row
        for row in records
        if len(row.get("visual_class_evidence", [])) == len(class_order)
        and len(row.get("semantic_class_evidence", [])) == len(class_order)
        and all(
            _is_finite_number(row.get(key))
            for key in (
                "visual_concentration",
                "semantic_concentration",
                "visual_semantic_disagreement",
            )
        )
        and all(
            _is_finite_number(value)
            for key in ("visual_class_evidence", "semantic_class_evidence")
            for value in row.get(key, [])
        )
    ]
    output = {
        "num_images": len(usable),
        "mean_visual_concentration": _safe_mean(
            [float(row["visual_concentration"]) for row in usable]
        ),
        "mean_semantic_concentration": _safe_mean(
            [float(row["semantic_concentration"]) for row in usable]
        ),
        "mean_visual_semantic_disagreement": _safe_mean(
            [float(row["visual_semantic_disagreement"]) for row in usable]
        ),
        "per_class": {},
    }
    for index, label in enumerate(class_order):
        output["per_class"][label] = {
            "mean_visual_evidence": _safe_mean(
                [float(row["visual_class_evidence"][index]) for row in usable]
            ),
            "mean_semantic_evidence": _safe_mean(
                [float(row["semantic_class_evidence"][index]) for row in usable]
            ),
        }
    return output


def selected_demonstration_statistics(records: list[dict]) -> dict[str, Any]:
    label_frequency: dict[str, int] = {}
    distinct_distribution: dict[str, int] = {}
    all_same = exactly_two_same = all_different = 0
    usable = 0
    for row in records:
        labels = [item["label"] for item in row.get("selected_demonstrations", [])]
        if len(labels) != 3:
            continue
        usable += 1
        for label in labels:
            label_frequency[label] = label_frequency.get(label, 0) + 1
        distinct = len(set(labels))
        key = str(distinct)
        distinct_distribution[key] = distinct_distribution.get(key, 0) + 1
        if distinct == 1:
            all_same += 1
        elif distinct == 2:
            exactly_two_same += 1
        elif distinct == 3:
            all_different += 1
    return {
        "num_images_with_three_demonstrations": usable,
        "selected_example_label_frequency": dict(sorted(label_frequency.items())),
        "selection_rule": "pure global Top-3; no visual anchor or class-balance constraint",
        "distinct_label_count_distribution": dict(sorted(distinct_distribution.items())),
        "three_examples_same_class_count": all_same,
        "exactly_two_examples_same_class_count": exactly_two_same,
        "three_examples_all_different_count": all_different,
    }


def _write_all_outputs(
    *,
    out_dir: Path,
    records: list[dict],
    class_order: list[str],
    expected_targets: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    embedding_preparation_seconds: float,
    invocation_wall_seconds: float,
    dataset_name: str,
    model_id: str,
    run_errors: list[dict],
    expected_target_ids: list[str],
    global_run_error_count: int = 0,
    formal_result_eligible: bool = True,
) -> None:
    _write_jsonl_atomic(out_dir / "predictions.jsonl", records)
    invalid_records = [row for row in records if row["invalid_output"]]
    _write_jsonl_atomic(out_dir / "invalid_outputs.jsonl", invalid_records)
    _write_jsonl_atomic(out_dir / "run_errors.jsonl", run_errors)
    unresolved_errors = unresolved_run_error_ids(
        records, run_errors, expected_target_ids
    )
    metrics, per_class, confusion = compute_summary(
        records,
        class_order,
        expected_targets,
        bootstrap_samples,
        bootstrap_seed,
        embedding_preparation_seconds,
        invocation_wall_seconds,
        unresolved_run_error_count=len(unresolved_errors) + global_run_error_count,
        historical_run_error_count=len(run_errors),
        formal_result_eligible=formal_result_eligible,
    )
    weight_stats = adaptive_weight_statistics(records)
    distribution_stats = class_distribution_statistics(records, class_order)
    demonstration_stats = selected_demonstration_statistics(records)
    _write_json_atomic(
        out_dir / "summary.json",
        {
            "metrics": metrics,
            "adaptive_weight_statistics": weight_stats,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json_atomic(out_dir / "adaptive_weight_statistics.json", weight_stats)
    _write_json_atomic(
        out_dir / "class_distribution_statistics.json", distribution_stats
    )
    _write_json_atomic(
        out_dir / "selected_demonstration_statistics.json", demonstration_stats
    )
    _write_csv_atomic(out_dir / "per_class_metrics.csv", per_class)
    _write_csv_atomic(out_dir / "confusion_matrix.csv", confusion)
    _write_confusion_matrix_png(out_dir / "confusion_matrix.png", confusion, class_order)
    final_row = {
        "Dataset": dataset_name,
        "Model": Path(model_id).name,
        "Method": "Adaptive RS-ViSemDS",
        "Accuracy": metrics["accuracy"],
        "Macro Precision": metrics["macro_precision"],
        "Macro Recall": metrics["macro_recall"],
        "Macro F1": metrics["macro_f1"],
        "Invalid Rate": metrics["invalid_output_rate"],
        "Time/Image": metrics[
            "average_total_time_including_amortized_embedding_seconds"
        ],
        "Run Complete": metrics["run_complete"],
        "Status": metrics["result_status"],
    }
    _write_csv_atomic(out_dir / "final_results.csv", [final_row])


def _write_checkpoint_outputs(
    out_dir: Path, records: list[dict], run_errors: list[dict]
) -> None:
    """Atomically persist every target without recomputing expensive statistics."""
    _write_jsonl_atomic(out_dir / "predictions.jsonl", records)
    _write_jsonl_atomic(
        out_dir / "invalid_outputs.jsonl",
        [row for row in records if row["invalid_output"]],
    )
    _write_jsonl_atomic(
        out_dir / "run_errors.jsonl",
        run_errors,
    )


def collect_environment_metadata(model_id: str, model_backend: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model_id": model_id,
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "numpy",
                "open-clip-torch",
                "Pillow",
            )
        },
        "git": _git_metadata(PROJECT_ROOT),
        "model_revision": None,
        "local_model_config_sha256": None,
    }
    model_path = Path(model_id).expanduser()
    config_path = model_path / "config.json"
    if config_path.is_file():
        metadata["local_model_config_sha256"] = sha256_file(config_path)
    if model_backend is not None:
        model_config = getattr(getattr(model_backend, "model", None), "config", None)
        metadata["model_revision"] = getattr(model_config, "_commit_hash", None)
    try:
        import torch

        metadata["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except (ImportError, RuntimeError, OSError) as exc:
        metadata["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return metadata


def write_environment_file(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status,
    }


def _write_confusion_matrix_png(
    path: Path,
    confusion_rows: list[dict],
    class_order: list[str],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    labels = [*class_order, INVALID_LABEL]
    cell = 72
    left = 220
    top = 160
    width = left + cell * len(labels) + 30
    height = top + cell * len(class_order) + 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    maximum = max(
        (int(row.get(label, 0)) for row in confusion_rows for label in labels),
        default=1,
    ) or 1
    for column, label in enumerate(labels):
        x = left + column * cell + cell // 2
        draw.text((x, top - 18), label, fill="black", font=font, anchor="ms")
    for row_index, (true_label, row) in enumerate(zip(class_order, confusion_rows)):
        y = top + row_index * cell
        draw.text((left - 10, y + cell // 2), true_label, fill="black", font=font, anchor="rm")
        for column, label in enumerate(labels):
            value = int(row.get(label, 0))
            intensity = value / maximum
            color = (
                int(245 - 155 * intensity),
                int(248 - 110 * intensity),
                255,
            )
            x = left + column * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline=(180, 180, 180))
            draw.text(
                (x + cell // 2, y + cell // 2),
                str(value),
                fill="black",
                font=font,
                anchor="mm",
            )
    draw.text((left, 20), "Confusion Matrix", fill="black", font=font)
    draw.text((20, top - 50), "True label", fill="black", font=font)
    draw.text((left, top - 50), "Predicted label", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def bootstrap_accuracy_ci(
    records: list[dict], samples: int, seed: int
) -> list[float] | None:
    if not records or samples <= 0:
        return None
    correctness = np.asarray(
        [float(bool(record["is_correct"])) for record in records], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = correctness[
            rng.integers(0, correctness.size, correctness.size)
        ].mean()
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _distribution_statistics(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            key: None
            for key in ("mean", "std", "min", "max", "median", "q25", "q75")
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def _mean_field(records: list[dict], field: str) -> float:
    return _safe_mean([float(row.get(field) or 0.0) for row in records])


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _infer_error_stage(record: dict) -> str:
    if not record["selected_demonstrations"]:
        if record["visual_concentration"] is None:
            return "retrieval_or_adaptive_weighting"
        return "demonstration_selection"
    if record["mllm_call_count"] == 0:
        return "prompt_construction"
    return "qwen_inference"


def _optional_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_path(path, PROJECT_ROOT)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
