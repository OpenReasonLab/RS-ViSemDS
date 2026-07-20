from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

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


INVALID_LABEL = "__INVALID__"
OFFICIAL_OPENAI_BASE = "https://api.openai.com/v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=("api", "transformers"), default="api")
    parser.add_argument(
        "--api-base",
        default=(
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("MLLM_API_BASE")
            or OFFICIAL_OPENAI_BASE
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("MLLM_API_KEY")
            or ""
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
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
    parser.add_argument(
        "--prompt-mode",
        choices=("minimal", "guided"),
        default="guided",
        help="Use only the candidate list (minimal) or include dataset-specific visual cues (guided).",
    )
    parser.add_argument(
        "--preserve-image-metadata",
        action="store_true",
        help="Send original image bytes. By default images are decoded and re-encoded as metadata-free PNGs.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def image_to_data_url(path: Path, preserve_metadata: bool = False) -> str:
    if not preserve_metadata:
        buffer = io.BytesIO()
        with Image.open(path) as image:
            clean_image = image.convert("RGB")
            clean_image.info.clear()
            clean_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"

    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


SYSTEM_PROMPT = (
    "You are a remote-sensing scene classification assistant. Analyze only visible "
    "content in the supplied overhead images. Do not use filenames, metadata, or "
    "unstated context. Select exactly one label from the candidate list."
)


DATASET_CONSIDERATIONS = {
    "aid": (
        "Consider the dominant land use, object geometry, spatial layout, road network, "
        "surface texture, and surrounding context. Distinguish Airport by runways, taxiways, "
        "or terminal-like structures; BaseballField by a diamond-shaped field; Beach by a "
        "sand-water boundary; Bridge by a narrow crossing structure over water or transport "
        "corridors; and Church by a distinctive worship-building footprint or tower. Carefully "
        "separate BareLand from Desert using patchiness, vegetation, development context, and "
        "terrain uniformity. Separate Center, Commercial, and DenseResidential using building "
        "scale, density, block organization, open plazas, parking, and road patterns."
    ),
    "nwpu_fg_urban": (
        "Focus on building density, building size and regularity, spacing, road structure, "
        "parking patterns, and transport infrastructure. Distinguish dense, medium, and sparse "
        "residential scenes by roof concentration, spacing, open space, and street layout. "
        "Mobile home parks usually contain many small, similarly oriented elongated units. "
        "Commercial areas tend to contain large buildings, access roads, and parking; industrial "
        "areas tend to contain large factory or warehouse roofs, yards, tanks, or service roads. "
        "Parking lots are dominated by marked parking surfaces and vehicles, while railway "
        "stations should show multiple tracks, platforms, or station-related structures."
    ),
}


def format_class_options(class_order: list[str]) -> str:
    return "\n".join(f"- {label}" for label in class_order)


def build_prompt(dataset: str, class_order: list[str], prompt_mode: str = "guided") -> str:
    classes = ", ".join(class_order)
    prompt = (
        "Task Instruction: Classify the target remote-sensing image into exactly one "
        "candidate class.\n\n"
        f"Candidate Label Set: {classes}\n"
        "Allowed answer strings must be copied exactly, including capitalization and underscores.\n\n"
        "Target Input: the supplied image is the unlabeled target image.\n\n"
    )
    if prompt_mode == "guided":
        considerations = DATASET_CONSIDERATIONS.get(
            dataset,
            "Consider the dominant land use, object geometry, spatial organization, texture, and context.",
        )
        prompt += f"Dataset-specific considerations: {considerations}\n\n"
    return prompt + (
        "Query: Classify the target image into exactly one candidate class.\n\n"
        "Output Format: Return exactly one compact JSON object and no other text: "
        '{"thoughts":"<brief observable visual evidence>",'
        '"answer":"<one candidate class>","score":<number from 0 to 1>}'
    )


def invalid_retry_instruction(class_order: list[str], previous_output: str = "") -> str:
    previous = previous_output.strip().replace("\n", " ")
    if len(previous) > 500:
        previous = previous[:500] + "..."
    return (
        "Your previous response could not be parsed as a valid prediction. "
        "Regenerate the answer now. Return exactly one compact JSON object and no other text. "
        "The answer value must be copied exactly from this allowed list:\n"
        f"{format_class_options(class_order)}\n"
        'Use this schema exactly: {"thoughts": "<brief visual evidence>", '
        '"answer": "<one allowed answer string>", "score": <number from 0 to 1>}\n'
        f"Previous invalid output: {previous}"
    )


def call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    image_path: Path,
    dataset: str,
    class_order: list[str],
    prompt_mode: str,
    preserve_image_metadata: bool,
    extra_instruction: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            build_prompt(dataset, class_order, prompt_mode)
                            + (("\n\n" + extra_instruction) if extra_instruction else "")
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path, preserve_image_metadata)
                        },
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(response, ensure_ascii=False)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return str(content)


def extract_json_object(raw: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_prediction(text: str, class_order: list[str]) -> dict:
    raw = text.strip()
    data = extract_json_object(raw)
    if data is not None:
        for key in ("answer", "prediction", "predicted_label", "label", "class"):
            value = data.get(key)
            if not isinstance(value, str):
                continue
            normalized = match_class(value, class_order)
            if normalized is not None:
                score = data.get("score", data.get("confidence", ""))
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    score = ""
                thoughts = data.get("thoughts", data.get("reason", ""))
                return {
                    "pred_label": normalized,
                    "raw_pred_label": value,
                    "parse_valid": 1,
                    "parse_mode": "json",
                    "score": score,
                    "thoughts": thoughts if isinstance(thoughts, str) else str(thoughts),
                }

    normalized = match_class(raw, class_order)
    if normalized is not None:
        return {
            "pred_label": normalized,
            "raw_pred_label": raw,
            "parse_valid": 1,
            "parse_mode": "exact_text",
            "score": "",
            "thoughts": "",
        }

    return {
        "pred_label": INVALID_LABEL,
        "raw_pred_label": raw,
        "parse_valid": 0,
        "parse_mode": "invalid",
        "score": "",
        "thoughts": "",
    }


def match_class(value: str, class_order: list[str]) -> str | None:
    value_norm = normalize_label(value)
    for cls in class_order:
        if normalize_label(cls) == value_norm:
            return cls
    found = []
    padded_value = f" {value_norm} "
    for cls in class_order:
        cls_norm = normalize_label(cls)
        if re.search(rf"(?<![a-z0-9]){re.escape(cls_norm)}(?![a-z0-9])", padded_value):
            found.append(cls)
    if len(found) == 1:
        return found[0]
    return None


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def usage_value(response: dict, *keys: str):
    usage = response.get("usage", {})
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return ""


def usage_json(response: dict) -> str:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return json.dumps(usage, ensure_ascii=False, sort_keys=True) if usage else ""


def mean_float(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return sum(values) / len(values) if values else 0.0


def sum_float(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows if row.get(key, "") != "")


def percentile_float(rows: list[dict], key: str, percentile: float) -> float:
    values = sorted(float(row[key]) for row in rows if row.get(key, "") != "")
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def int_value(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def calibration_metrics(rows: list[dict], num_bins: int = 10) -> dict:
    scored = []
    for row in rows:
        try:
            score = float(row.get("score", ""))
        except (TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0:
            scored.append((score, int_value(row.get("correct"))))
    if not scored:
        return {"num_scored_predictions": 0, "brier_score": None, "ece_10_bin": None}

    brier = sum((score - correct) ** 2 for score, correct in scored) / len(scored)
    ece = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        members = [
            (score, correct) for score, correct in scored
            if lower <= score < upper or (bin_index == num_bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        avg_score = sum(score for score, _ in members) / len(members)
        avg_accuracy = sum(correct for _, correct in members) / len(members)
        ece += len(members) / len(scored) * abs(avg_accuracy - avg_score)
    return {
        "num_scored_predictions": len(scored),
        "brier_score": brier,
        "ece_10_bin": ece,
    }


def bootstrap_accuracy_ci(rows: list[dict], samples: int, seed: int) -> list[float] | None:
    if not rows or samples <= 0:
        return None
    values = [int_value(row.get("correct")) for row in rows]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    estimates.sort()
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples))
    return [estimates[lower_index], estimates[upper_index]]


def summarize_rows(
    rows: list[dict],
    class_order: list[str],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
):
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    metrics = {}
    per_class_rows = []
    matrices = {}
    class_to_idx = {label: index for index, label in enumerate(class_order)}

    for model, model_rows in by_model.items():
        matrix = [[0 for _ in range(len(class_order) + 1)] for _ in class_order]
        for row in model_rows:
            true_idx = class_to_idx[row["true_label"]]
            pred_idx = class_to_idx.get(row.get("pred_label", ""), len(class_order))
            matrix[true_idx][pred_idx] += 1

        precision_values = []
        recall_values = []
        f1_values = []
        per_class_accuracy = {}
        for label, index in class_to_idx.items():
            tp = matrix[index][index]
            support = sum(matrix[index])
            pred_count = sum(matrix[row_index][index] for row_index in range(len(class_order)))
            recall = tp / support if support else 0.0
            precision = tp / pred_count if pred_count else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            per_class_accuracy[label] = recall
            per_class_rows.append({
                "model": model,
                "class": label,
                "support": support,
                "accuracy": recall,
                "precision": precision,
                "f1": f1,
                "invalid_predictions": matrix[index][-1],
            })

        total = len(model_rows)
        total_predict_seconds = sum_float(model_rows, "predict_seconds")
        total_recorded_seconds = sum_float(model_rows, "total_seconds")
        prompt_tokens = sum(int_value(row.get("prompt_tokens")) for row in model_rows)
        completion_tokens = sum(int_value(row.get("completion_tokens")) for row in model_rows)
        total_tokens = sum(int_value(row.get("total_tokens")) for row in model_rows)
        image_tokens = sum(int_value(row.get("image_tokens")) for row in model_rows)
        valid_predictions = sum(
            1 for row in model_rows
            if not row.get("error") and int_value(row.get("parse_valid")) == 1
        )
        total_attempts = sum(int_value(row.get("attempt_count")) for row in model_rows)
        model_metrics = {
            "num_targets": total,
            "num_correct": sum(int_value(row.get("correct")) for row in model_rows),
            "overall_accuracy": (
                sum(int_value(row.get("correct")) for row in model_rows) / total if total else 0.0
            ),
            "macro_precision": (
                sum(precision_values) / len(precision_values) if precision_values else 0.0
            ),
            "macro_recall": (
                sum(recall_values) / len(recall_values) if recall_values else 0.0
            ),
            "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
            "num_valid_predictions": valid_predictions,
            "valid_prediction_rate": valid_predictions / total if total else 0.0,
            "total_predict_seconds": total_predict_seconds,
            "total_recorded_seconds": total_recorded_seconds,
            "avg_predict_seconds_per_target": mean_float(model_rows, "predict_seconds"),
            "avg_total_seconds_per_target": mean_float(model_rows, "total_seconds"),
            "median_predict_seconds_per_target": percentile_float(
                model_rows, "predict_seconds", 0.5
            ),
            "p95_predict_seconds_per_target": percentile_float(
                model_rows, "predict_seconds", 0.95
            ),
            "throughput_targets_per_minute": (
                60.0 * total / total_recorded_seconds if total_recorded_seconds else 0.0
            ),
            "total_request_attempts": total_attempts,
            "total_retries": max(0, total_attempts - total),
            "num_parse_failures": sum(
                1 for row in model_rows
                if not row.get("error") and int_value(row.get("parse_valid")) == 0
            ),
            "num_request_errors": sum(1 for row in model_rows if row.get("error")),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "image_tokens": image_tokens,
            "avg_prompt_tokens_per_target": prompt_tokens / total if total else 0.0,
            "avg_completion_tokens_per_target": completion_tokens / total if total else 0.0,
            "avg_total_tokens_per_target": total_tokens / total if total else 0.0,
            "avg_image_tokens_per_target": image_tokens / total if total else 0.0,
            "per_class_accuracy": per_class_accuracy,
        }
        model_metrics.update(calibration_metrics(model_rows))
        model_metrics["overall_accuracy_bootstrap_95_ci"] = bootstrap_accuracy_ci(
            model_rows, bootstrap_samples, bootstrap_seed
        )
        model_metrics["bootstrap_samples"] = bootstrap_samples
        metrics[model] = model_metrics
        matrices[model] = matrix
    return metrics, per_class_rows, matrices


def write_outputs(
    out_dir: Path,
    rows: list[dict],
    class_order: list[str],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
    expected_targets: int | None = None,
    invocation_wall_seconds: float | None = None,
) -> None:
    pred_fields = [
        "dataset", "strategy", "shot", "model", "target_id", "target_path",
        "true_label", "pred_label", "correct", "score", "thoughts",
        "parse_valid", "parse_mode", "raw_pred_label", "raw_response", "error",
        "response_model", "response_id", "system_fingerprint", "request_id",
        "attempt_count", "prompt_tokens", "completion_tokens", "total_tokens",
        "image_tokens", "usage_json",
        "predict_seconds", "total_seconds",
    ]
    write_csv(out_dir / "predictions.csv", rows, pred_fields)

    metrics, per_class_rows, matrices = summarize_rows(
        rows, class_order, bootstrap_samples, bootstrap_seed
    )
    parse_failures = sum(
        1 for row in rows if not row.get("error") and int_value(row.get("parse_valid")) == 0
    )
    request_errors = sum(1 for row in rows if row.get("error"))
    write_json(out_dir / "summary.json", {
        "metrics": metrics,
        "num_parse_failures": parse_failures,
        "num_request_errors": request_errors,
        "num_completed_targets": len(rows),
        "num_expected_targets": expected_targets,
        "completion_rate": (
            len(rows) / expected_targets if expected_targets else None
        ),
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
            row.update({label: value for label, value in zip(pred_columns, values)})
            matrix_rows.append(row)
        write_csv(
            out_dir / f"confusion_matrix_{model_name}.csv",
            matrix_rows,
            ["true_label", *pred_columns],
        )


def format_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception:
        detail = ""
    return f"HTTP {exc.code}: {detail or exc.reason}"


def main():
    args = parse_args()
    invocation_start = time.perf_counter()
    if args.backend == "api" and not args.api_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY or pass --api-key."
        )
    manifest_dir = repo_path(args.manifest_dir, Path.cwd())
    out_dir = repo_path(args.out_dir, Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(manifest_dir / "summary.json")
    class_order = read_json(manifest_dir / "class_order.json")["classes"]
    data_root = repo_path(summary["data_root"], Path.cwd())
    evaluation = read_csv(manifest_dir / "evaluation.csv")
    if args.limit is not None:
        evaluation = evaluation[:args.limit]

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
        "manifest_dir": args.manifest_dir,
        "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        "support_sha256": sha256_file(manifest_dir / "support.csv"),
        "class_order_sha256": sha256_file(manifest_dir / "class_order.json"),
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
        "timeout": args.timeout,
        "retries": args.retries,
        "retry_delay": args.retry_delay,
        "max_consecutive_request_errors": args.max_consecutive_request_errors,
        "invalid_retries": args.invalid_retries,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "prompt_mode": args.prompt_mode,
        "image_preprocessing": (
            "original_bytes_with_metadata"
            if args.preserve_image_metadata
            else "decoded_rgb_metadata_free_png"
        ),
        "system_prompt": SYSTEM_PROMPT,
        "dataset_considerations": (
            DATASET_CONSIDERATIONS.get(summary["dataset"], "")
            if args.prompt_mode == "guided"
            else ""
        ),
        "prompt": build_prompt(summary["dataset"], class_order, args.prompt_mode),
        "candidate_classes": class_order,
    }
    config_path = out_dir / "run_config.json"
    if args.resume and config_path.exists():
        old = read_json(config_path)
        for key in (
            "dataset", "manifest_dir", "evaluation_sha256", "support_sha256",
            "class_order_sha256", "model", "backend", "prompt_mode",
            "image_preprocessing", "local_mllm_implementation_version",
            "conversation_context", "internvl_flash_attention_policy",
            "internvl_flash_attention_enabled",
        ):
            if old.get(key) != config.get(key):
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"{old.get(key)!r} != {config.get(key)!r}"
                )
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
    consecutive_request_errors = 0
    aborted_for_errors = False

    for target in evaluation:
        if target["target_id"] in completed_ids:
            print(f"Skipping completed target: {target['target_id']}")
            continue

        image_path = data_root / target["path"]
        total_start = time.perf_counter()
        predict_start = time.perf_counter()
        error = ""
        raw_text = ""
        response = {}
        parsed = {
            "pred_label": INVALID_LABEL,
            "raw_pred_label": "",
            "parse_valid": 0,
            "parse_mode": "invalid",
            "score": "",
            "thoughts": "",
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
                    messages = [{
                        "role": "user",
                        "content": [
                            text_part(
                                SYSTEM_PROMPT
                                + "\n\n"
                                + build_prompt(summary["dataset"], class_order, args.prompt_mode)
                                + (("\n\n" + invalid_retry_note) if invalid_retry_note else "")
                            ),
                            image_part(),
                        ],
                    }]
                    raw_text = local_model.generate_from_messages(
                        messages, [load_rgb_image(image_path)]
                    )
                    response = {"model": args.model, "backend": "transformers"}
                else:
                    response = call_openai_compatible(
                        args.api_base,
                        args.api_key,
                        args.model,
                        image_path,
                        summary["dataset"],
                        class_order,
                        args.prompt_mode,
                        args.preserve_image_metadata,
                        invalid_retry_note,
                        args.temperature,
                        args.max_tokens,
                        args.timeout,
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

        predict_seconds = time.perf_counter() - predict_start
        total_seconds = time.perf_counter() - total_start
        if error:
            parsed["pred_label"] = INVALID_LABEL
            parsed["parse_valid"] = 0
            parsed["parse_mode"] = "request_error"
        correct = int(
            not error
            and parsed["parse_valid"] == 1
            and parsed["pred_label"] == target["label"]
        )

        row = {
            "dataset": summary["dataset"],
            "strategy": "zero_shot",
            "shot": 0,
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
            "response_model": response.get("model", "") if isinstance(response, dict) else "",
            "response_id": response.get("id", "") if isinstance(response, dict) else "",
            "system_fingerprint": (
                response.get("system_fingerprint", "") if isinstance(response, dict) else ""
            ),
            "request_id": (
                response.get("request_id", response.get("request-id", ""))
                if isinstance(response, dict)
                else ""
            ),
            "attempt_count": attempt_count,
            "prompt_tokens": usage_value(response, "prompt_tokens", "input_tokens"),
            "completion_tokens": usage_value(response, "completion_tokens", "output_tokens"),
            "total_tokens": usage_value(response, "total_tokens"),
            "image_tokens": usage_value(response, "image_tokens"),
            "usage_json": usage_json(response),
            "predict_seconds": f"{predict_seconds:.4f}",
            "total_seconds": f"{total_seconds:.4f}",
        }
        rows.append(row)
        completed_ids.add(target["target_id"])
        if error:
            consecutive_request_errors += 1
        else:
            consecutive_request_errors = 0
        print(
            f"{target['target_id']}: true={target['label']} "
            f"pred={parsed['pred_label']} correct={correct} "
            f"parse={parsed['parse_mode']} error={bool(error)}"
        )
        if len(rows) % args.save_every == 0:
            write_outputs(
                out_dir,
                rows,
                class_order,
                expected_targets=len(evaluation),
                invocation_wall_seconds=time.perf_counter() - invocation_start,
            )
        if (
            args.max_consecutive_request_errors > 0
            and consecutive_request_errors >= args.max_consecutive_request_errors
        ):
            aborted_for_errors = True
            print(
                "Aborting after "
                f"{consecutive_request_errors} consecutive request errors.",
                flush=True,
            )
            break

    write_outputs(
        out_dir,
        rows,
        class_order,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_targets=len(evaluation),
        invocation_wall_seconds=time.perf_counter() - invocation_start,
    )
    if aborted_for_errors:
        raise SystemExit(
            "Experiment aborted because the API repeatedly rejected or failed requests. "
            f"See {out_dir / 'predictions.csv'}"
        )
    print(f"Wrote zero-shot MLLM results to {out_dir}")


if __name__ == "__main__":
    main()
