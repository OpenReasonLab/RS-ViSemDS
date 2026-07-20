from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from strict_fewshot.local_mllm import (
    LOCAL_MLLM_IMPLEMENTATION_VERSION,
    TransformersVisionLLM,
    image_part,
    load_rgb_image,
    text_part,
)
from strict_fewshot.utils import (
    read_csv,
    read_json,
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)
from run_zero_shot_mllm import (
    DATASET_CONSIDERATIONS,
    INVALID_LABEL,
    SYSTEM_PROMPT,
    extract_text,
    format_class_options,
    format_http_error,
    image_to_data_url,
    invalid_retry_instruction,
    int_value,
    parse_prediction,
    summarize_rows,
    usage_json,
    usage_value,
)


OFFICIAL_OPENAI_BASE = "https://api.openai.com/v1"


class ProviderResponseError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate query-specific total-shot random MLLM prompts."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--examples-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=("api", "transformers"), default="api")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or os.environ.get("MLLM_API_BASE") or OFFICIAL_OPENAI_BASE,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("MLLM_API_KEY") or "",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument(
        "--local-image-mode",
        choices=("native_multi",),
        default="native_multi",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-consecutive-request-errors", type=int, default=3)
    parser.add_argument(
        "--invalid-retries",
        type=int,
        default=0,
        help="Optional non-paper retry count; the manuscript protocol uses 0.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--prompt-mode", choices=("minimal", "guided"), default="minimal")
    parser.add_argument("--preserve-image-metadata", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_task_prompt(
    dataset: str,
    class_order: list[str],
    shot: int,
    prompt_mode: str,
) -> str:
    prompt = (
        "Task Instruction: Use the randomly sampled labeled images below as visual "
        "reference examples to classify the target image.\n\n"
        f"Candidate Label Set: {', '.join(class_order)}\n"
        "Allowed answer strings must be copied exactly, including capitalization and underscores.\n\n"
        f"Randomly Sampled Demonstrations: {shot} labeled image(s) follow.\n\n"
    )
    if prompt_mode == "guided":
        prompt += (
            "Dataset-specific considerations: "
            + DATASET_CONSIDERATIONS.get(dataset, "Inspect scene layout, objects, texture, and context.")
            + "\n\n"
        )
    return prompt + (
        "Query: Classify the target image into exactly one candidate class.\n\n"
        "Output Format: Return exactly one compact JSON object and no other text: "
        '{"thoughts":"<brief observable visual evidence>",'
        '"answer":"<one candidate class>","score":<number from 0 to 1>}'
    )


def build_content(
    data_root: Path,
    target_path: str,
    dataset: str,
    class_order: list[str],
    examples: list[dict],
    prompt_mode: str,
    preserve_image_metadata: bool,
) -> list[dict]:
    content = [{
        "type": "text",
        "text": build_task_prompt(
            dataset,
            class_order,
            len(examples),
            prompt_mode,
        ),
    }]
    for index, example in enumerate(examples, start=1):
        content.extend([
            {
                "type": "text",
                "text": f"Labeled example {index}/{len(examples)}. Ground-truth label: {example['example_label']}",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        data_root / example["example_path"], preserve_image_metadata
                    )
                },
            },
        ])
    content.extend([
        {
            "type": "text",
            "text": "Target Input: the next image is the unlabeled target image.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(data_root / target_path, preserve_image_metadata)
            },
        },
    ])
    return content


def build_local_messages_and_images(
    data_root: Path,
    target_path: str,
    dataset: str,
    class_order: list[str],
    examples: list[dict],
    prompt_mode: str,
    model_id: str,
    local_image_mode: str,
) -> tuple[list[dict], list]:
    loaded_images = [load_rgb_image(data_root / example["example_path"]) for example in examples]
    loaded_images.append(load_rgb_image(data_root / target_path))

    content = [
        text_part(
            SYSTEM_PROMPT + "\n\n" + build_task_prompt(
                dataset,
                class_order,
                len(examples),
                prompt_mode,
            )
        )
    ]
    for index, example in enumerate(examples, start=1):
        content.extend([
            text_part(
                f"Labeled example {index}/{len(examples)}. "
                f"Ground-truth label: {example['example_label']}"
            ),
            image_part(),
        ])
    content.extend([
        text_part("Target Input: the next image is the unlabeled target image."),
        image_part(),
    ])
    return [{"role": "user", "content": content}], loaded_images


def call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    content: list[dict],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if isinstance(result, dict) and result.get("error"):
        raise ProviderResponseError(
            "Provider returned an application error: "
            + json.dumps(result["error"], ensure_ascii=False)
        )
    if not isinstance(result, dict) or not result.get("choices"):
        raise ProviderResponseError(
            "Provider response has no choices: "
            + json.dumps(result, ensure_ascii=False)[:1000]
        )
    return result


def load_and_validate_examples(
    examples_csv: Path,
    evaluation: list[dict],
    support: list[dict],
) -> tuple[dict[str, list[dict]], int]:
    rows = read_csv(examples_csv)
    if not rows:
        raise ValueError(f"No rows in examples CSV: {examples_csv}")
    shots = {int(row["shot"]) for row in rows}
    strategies = {row["strategy"] for row in rows}
    if len(shots) != 1 or strategies != {"random"}:
        raise ValueError(f"Expected one random shot value, got shots={shots}, strategies={strategies}")
    shot = next(iter(shots))
    eval_by_id = {row["target_id"]: row for row in evaluation}
    support_pairs = {(row["label"], row["path"]) for row in support}
    support_paths = {row["path"] for row in support}
    evaluation_paths = {row["path"] for row in evaluation}
    if support_paths & evaluation_paths:
        raise ValueError("Evaluation and support manifests overlap by image path.")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        target = eval_by_id.get(row["target_id"])
        if target is None:
            raise ValueError(f"Unknown target_id in examples CSV: {row['target_id']}")
        if row["target_path"] != target["path"] or row["target_label"] != target["label"]:
            raise ValueError(f"Target metadata mismatch for {row['target_id']}")
        if (row["example_label"], row["example_path"]) not in support_pairs:
            raise ValueError(f"Example is not in support pool: {row['example_path']}")
        grouped.setdefault(row["target_id"], []).append(row)

    for target in evaluation:
        selected = sorted(grouped.get(target["target_id"], []), key=lambda row: int(row["rank"]))
        if len(selected) != shot:
            raise ValueError(
                f"Target {target['target_id']} has {len(selected)} examples, expected {shot}"
            )
        ranks = [int(row["rank"]) for row in selected]
        paths = [row["example_path"] for row in selected]
        if ranks != list(range(1, shot + 1)):
            raise ValueError(f"Non-contiguous ranks for target {target['target_id']}: {ranks}")
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate random example for target {target['target_id']}")
        grouped[target["target_id"]] = selected
    return grouped, shot


def write_outputs(
    out_dir: Path,
    rows: list[dict],
    class_order: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    expected_targets: int,
    invocation_wall_seconds: float,
) -> None:
    fields = [
        "dataset", "strategy", "shot", "model", "target_id", "target_path",
        "true_label", "pred_label", "correct", "score", "thoughts",
        "parse_valid", "parse_mode", "raw_pred_label", "raw_response", "error",
        "num_examples", "example_labels", "example_paths", "example_ranks", "sampling_seed",
        "response_model", "response_id", "system_fingerprint", "request_id",
        "attempt_count", "prompt_tokens", "completion_tokens", "total_tokens", "image_tokens",
        "usage_json", "predict_seconds", "total_seconds",
    ]
    write_csv(out_dir / "predictions.csv", rows, fields)
    metrics, per_class_rows, matrices = summarize_rows(
        rows, class_order, bootstrap_samples, bootstrap_seed
    )
    write_json(out_dir / "summary.json", {
        "metrics": metrics,
        "num_parse_failures": sum(
            1 for row in rows if not row.get("error") and int_value(row.get("parse_valid")) == 0
        ),
        "num_request_errors": sum(1 for row in rows if row.get("error")),
        "num_completed_targets": len(rows),
        "num_expected_targets": expected_targets,
        "completion_rate": len(rows) / expected_targets if expected_targets else None,
        "invocation_wall_seconds": invocation_wall_seconds,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_csv(
        out_dir / "per_class_accuracy.csv",
        per_class_rows,
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
            out_dir / f"confusion_matrix_{model_name}.csv",
            matrix_rows,
            ["true_label", *pred_columns],
        )


def main():
    args = parse_args()
    invocation_start = time.perf_counter()
    if args.backend == "api" and not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    manifest_dir = repo_path(args.manifest_dir, Path.cwd())
    examples_csv = repo_path(args.examples_csv, Path.cwd())
    out_dir = repo_path(args.out_dir, Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(manifest_dir / "summary.json")
    class_order = read_json(manifest_dir / "class_order.json")["classes"]
    data_root = repo_path(summary["data_root"], Path.cwd())
    full_evaluation = read_csv(manifest_dir / "evaluation.csv")
    support = read_csv(manifest_dir / "support.csv")
    examples_by_target, shot = load_and_validate_examples(
        examples_csv, full_evaluation, support
    )
    evaluation = full_evaluation[:args.limit] if args.limit is not None else full_evaluation
    sampling_config_path = examples_csv.parent / "random_sampling_config.json"
    sampling_config = read_json(sampling_config_path) if sampling_config_path.exists() else {}

    local_model = None
    if args.backend == "transformers":
        local_model = TransformersVisionLLM(
            args.model,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_new_tokens=args.max_tokens,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": (
            "local_transformers"
            if args.backend == "transformers"
            else "openai" if args.api_base.rstrip("/") == OFFICIAL_OPENAI_BASE else "openai_compatible"
        ),
        "dataset": summary["dataset"],
        "strategy": "random",
        "shot": shot,
        "shot_definition": "query_specific_total_labeled_images",
        "seed": sampling_config.get("seed"),
        "manifest_dir": args.manifest_dir,
        "examples_csv": args.examples_csv,
        "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        "support_sha256": sha256_file(manifest_dir / "support.csv"),
        "class_order_sha256": sha256_file(manifest_dir / "class_order.json"),
        "examples_sha256": sha256_file(examples_csv),
        "sampling_config_sha256": (
            sha256_file(sampling_config_path) if sampling_config_path.exists() else None
        ),
        "num_requested_targets": len(evaluation),
        "model": args.model,
        "backend": args.backend,
        "local_mllm_implementation_version": (
            LOCAL_MLLM_IMPLEMENTATION_VERSION if args.backend == "transformers" else None
        ),
        "conversation_context": "stateless_new_context_per_target",
        "internvl_flash_attention_policy": "auto_detect",
        "internvl_flash_attention_enabled": (
            getattr(local_model, "use_flash_attn", None) if local_model is not None else None
        ),
        "api_format": "openai" if args.backend == "api" else None,
        "api_base": args.api_base,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "torch_dtype": args.torch_dtype if args.backend == "transformers" else None,
        "device_map": args.device_map if args.backend == "transformers" else None,
        "min_pixels": args.min_pixels if args.backend == "transformers" else None,
        "max_pixels": args.max_pixels if args.backend == "transformers" else None,
        "local_image_mode": args.local_image_mode if args.backend == "transformers" else None,
        "timeout": args.timeout,
        "retries": args.retries,
        "retry_delay": args.retry_delay,
        "invalid_retries": args.invalid_retries,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "prompt_mode": args.prompt_mode,
        "image_preprocessing": (
            "original_bytes_with_metadata"
            if args.preserve_image_metadata
            else "decoded_rgb_metadata_free_png"
        ),
        "candidate_classes": class_order,
        "system_prompt": SYSTEM_PROMPT,
        "prompt": build_task_prompt(summary["dataset"], class_order, shot, args.prompt_mode),
    }
    config_path = out_dir / "run_config.json"
    if args.resume and config_path.exists():
        old = read_json(config_path)
        for key in (
            "dataset", "strategy", "shot", "model", "backend",
            "examples_csv", "prompt_mode", "image_preprocessing", "local_image_mode",
            "local_mllm_implementation_version",
            "conversation_context", "internvl_flash_attention_policy",
            "internvl_flash_attention_enabled",
            "evaluation_sha256", "support_sha256", "class_order_sha256",
            "examples_sha256", "sampling_config_sha256",
        ):
            if old.get(key) != config.get(key):
                raise ValueError(f"Resume configuration mismatch for {key}: {old.get(key)!r} != {config.get(key)!r}")
    write_json(config_path, config)

    predictions_path = out_dir / "predictions.csv"
    existing_rows = read_csv(predictions_path) if args.resume and predictions_path.exists() else []
    rows = [
        row for row in existing_rows
        if not row.get("error") and int_value(row.get("parse_valid")) == 1
    ]
    discarded_rows = len(existing_rows) - len(rows)
    if discarded_rows:
        print(
            f"Resume discarded {discarded_rows} failed/invalid row(s); "
            "those targets will be attempted again."
        )
    completed_ids = {row["target_id"] for row in rows}
    consecutive_errors = 0

    for target in evaluation:
        if target["target_id"] in completed_ids:
            print(f"Skipping completed target: {target['target_id']}")
            continue
        examples = examples_by_target[target["target_id"]]
        start = time.perf_counter()
        response = {}
        raw_text = ""
        error = ""
        parsed = {
            "pred_label": INVALID_LABEL, "raw_pred_label": "", "parse_valid": 0,
            "parse_mode": "invalid", "score": "", "thoughts": "",
        }
        attempt_count = 0
        invalid_retry_note = ""
        invalid_regens_used = 0
        request_attempts = 1 if args.backend == "transformers" else args.retries + 1
        attempts = request_attempts + max(0, args.invalid_retries)
        for attempt_count in range(1, attempts + 1):
            try:
                if args.backend == "transformers":
                    assert local_model is not None
                    messages, images = build_local_messages_and_images(
                        data_root, target["path"], summary["dataset"], class_order,
                        examples, args.prompt_mode, args.model, args.local_image_mode,
                    )
                    if invalid_retry_note:
                        messages[0]["content"].append(text_part(invalid_retry_note))
                    raw_text = local_model.generate_from_messages(messages, images)
                    response = {"model": args.model, "backend": "transformers"}
                else:
                    content = build_content(
                        data_root, target["path"], summary["dataset"], class_order,
                        examples, args.prompt_mode, args.preserve_image_metadata,
                    )
                    if invalid_retry_note:
                        content.append({"type": "text", "text": invalid_retry_note})
                    response = call_openai_compatible(
                        args.api_base, args.api_key, args.model, content,
                        args.temperature, args.max_tokens, args.timeout,
                    )
                    raw_text = extract_text(response)
                parsed = parse_prediction(raw_text, class_order)
                error = ""
                if parsed["parse_valid"] == 1:
                    break
                if invalid_regens_used < max(0, args.invalid_retries):
                    invalid_regens_used += 1
                    invalid_retry_note = invalid_retry_instruction(class_order, raw_text)
                    print(
                        f"{target['target_id']}: invalid parse, regenerating "
                        f"({invalid_regens_used}/{args.invalid_retries})"
                    )
                    continue
                break
            except urllib.error.HTTPError as exc:
                error = format_http_error(exc)
                if exc.code not in {408, 429} and not 500 <= exc.code < 600:
                    break
            except ProviderResponseError as exc:
                error = str(exc)
                break
            except (
                RuntimeError,
                ValueError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                OSError,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"
            if args.backend == "api" and attempt_count <= args.retries:
                time.sleep(args.retry_delay * attempt_count)

        elapsed = time.perf_counter() - start
        if error:
            parsed.update({"pred_label": INVALID_LABEL, "parse_valid": 0, "parse_mode": "request_error"})
        correct = int(
            not error and parsed["parse_valid"] == 1 and parsed["pred_label"] == target["label"]
        )
        row = {
            "dataset": summary["dataset"], "strategy": "random", "shot": shot,
            "model": args.model, "target_id": target["target_id"], "target_path": target["path"],
            "true_label": target["label"], "pred_label": parsed["pred_label"], "correct": correct,
            "score": parsed["score"], "thoughts": parsed["thoughts"],
            "parse_valid": parsed["parse_valid"], "parse_mode": parsed["parse_mode"],
            "raw_pred_label": parsed["raw_pred_label"], "raw_response": raw_text, "error": error,
            "num_examples": len(examples),
            "example_labels": "|".join(example["example_label"] for example in examples),
            "example_paths": "|".join(example["example_path"] for example in examples),
            "example_ranks": "|".join(example["rank"] for example in examples),
            "sampling_seed": examples[0].get("sampling_seed", ""),
            "response_model": response.get("model", "") if isinstance(response, dict) else "",
            "response_id": response.get("id", "") if isinstance(response, dict) else "",
            "system_fingerprint": response.get("system_fingerprint", "") if isinstance(response, dict) else "",
            "request_id": response.get("request_id", response.get("request-id", "")) if isinstance(response, dict) else "",
            "attempt_count": attempt_count,
            "prompt_tokens": usage_value(response, "prompt_tokens", "input_tokens"),
            "completion_tokens": usage_value(response, "completion_tokens", "output_tokens"),
            "total_tokens": usage_value(response, "total_tokens"),
            "image_tokens": usage_value(response, "image_tokens"),
            "usage_json": usage_json(response),
            "predict_seconds": f"{elapsed:.4f}", "total_seconds": f"{elapsed:.4f}",
        }
        rows.append(row)
        completed_ids.add(target["target_id"])
        consecutive_errors = consecutive_errors + 1 if error else 0
        print(
            f"{target['target_id']}: shot={shot} true={target['label']} "
            f"pred={parsed['pred_label']} correct={correct} parse={parsed['parse_mode']} error={bool(error)}"
        )
        if len(rows) % args.save_every == 0:
            write_outputs(
                out_dir, rows, class_order, args.bootstrap_samples, args.bootstrap_seed,
                len(evaluation), time.perf_counter() - invocation_start,
            )
        if args.max_consecutive_request_errors > 0 and consecutive_errors >= args.max_consecutive_request_errors:
            print(f"Stopping after {consecutive_errors} consecutive request errors.")
            break

    write_outputs(
        out_dir, rows, class_order, args.bootstrap_samples, args.bootstrap_seed,
        len(evaluation), time.perf_counter() - invocation_start,
    )
    print(f"Wrote random few-shot MLLM results to {out_dir}")


if __name__ == "__main__":
    main()
