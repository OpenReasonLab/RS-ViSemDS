from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from strict_fewshot.data import FewShotImageDataset
from strict_fewshot.models import (
    build_model,
    classification_head,
    set_head_train_mode,
    trainable_head_parameters,
)
from strict_fewshot.utils import (
    read_csv,
    read_json,
    runtime_metadata,
    sha256_file,
    write_csv,
    write_json,
)


MODEL_DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit_tiny": "ViT-Tiny",
    "vit_small": "ViT-Small",
}


class RandomQuarterTurn:
    def __call__(self, image: Image.Image) -> Image.Image:
        return TF.rotate(image, 90 * random.randrange(4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free full-data linear-probe baseline on a fixed evaluation manifest."
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "resnet50", "vit_tiny", "vit_small"],
        choices=list(MODEL_DISPLAY_NAMES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int, choices=[10], default=10)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-images", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Smoke-test only: cap support and evaluation rows per class.",
    )
    return parser.parse_args()


def resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    try:
        import numpy as np

        np.random.seed(worker_seed + worker_id)
    except ImportError:
        pass


def make_transforms():
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomQuarterTurn(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform


def as_dataset_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"example_label": row["label"], "example_path": row["path"]}
        for row in rows
    ]


def cap_per_class(
    rows: list[dict[str, str]], class_order: list[str], limit: int
) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    return [row for label in class_order for row in grouped[label][:limit]]


def stratified_support_split(
    support_rows: list[dict[str, str]],
    class_order: list[str],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, int]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in support_rows:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    training_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = {}
    for label in class_order:
        rows = list(grouped[label])
        if len(rows) < 2:
            raise ValueError(f"Class {label!r} needs at least two support images.")
        rng.shuffle(rows)
        validation_count = max(1, int(round(len(rows) * validation_ratio)))
        validation_count = min(validation_count, len(rows) - 1)
        validation_rows.extend(rows[:validation_count])
        training_rows.extend(rows[validation_count:])
        counts[label] = {
            "support": len(rows),
            "selection_train": len(rows) - validation_count,
            "selection_validation": validation_count,
        }
    rng.shuffle(training_rows)
    rng.shuffle(validation_rows)
    return training_rows, validation_rows, counts


def assert_manifest_is_leakage_free(
    manifest_summary: dict,
    support_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
    class_order: list[str],
    smoke_limit: int,
) -> None:
    if manifest_summary.get("split_mode") != "fixed_per_class":
        raise ValueError("This runner requires a fixed_per_class manifest.")
    if smoke_limit <= 0 and int(manifest_summary.get("eval_per_class", 0)) != 100:
        raise ValueError("This experiment requires exactly 100 evaluation images per class.")

    support_paths = {row["path"] for row in support_rows}
    evaluation_paths = {row["path"] for row in evaluation_rows}
    overlap = support_paths & evaluation_paths
    if overlap:
        raise ValueError(f"Support/evaluation leakage detected in {len(overlap)} paths.")

    expected = set(class_order)
    if {row["label"] for row in support_rows} != expected:
        raise ValueError("Support labels do not exactly match class_order.json.")
    if {row["label"] for row in evaluation_rows} != expected:
        raise ValueError("Evaluation labels do not exactly match class_order.json.")

    if smoke_limit <= 0:
        evaluation_counts = defaultdict(int)
        for row in evaluation_rows:
            evaluation_counts[row["label"]] += 1
        wrong = {label: evaluation_counts[label] for label in class_order if evaluation_counts[label] != 100}
        if wrong:
            raise ValueError(f"Evaluation class counts are not 100: {wrong}")


def make_loader(
    rows: list[dict[str, str]],
    data_root: Path,
    class_to_idx: dict[str, int],
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = FewShotImageDataset(as_dataset_rows(rows), data_root, class_to_idx, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def confusion_metrics(
    true_indices: list[int], predicted_indices: list[int], num_classes: int
) -> tuple[dict[str, float], list[list[int]], list[dict[str, float]]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_index, predicted_index in zip(true_indices, predicted_indices):
        matrix[true_index][predicted_index] += 1

    per_class = []
    for index in range(num_classes):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total = len(true_indices)
    correct = sum(matrix[i][i] for i in range(num_classes))
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": statistics.fmean(row["precision"] for row in per_class),
        "macro_recall": statistics.fmean(row["recall"] for row in per_class),
        "macro_f1": statistics.fmean(row["f1"] for row in per_class),
    }
    return metrics, matrix, per_class


def train_one_epoch(
    model,
    model_name: str,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    set_head_train_mode(model, model_name)
    loss_sum = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
    return loss_sum / sample_count


@torch.inference_mode()
def evaluate_loader(model, loader: DataLoader, criterion, device: torch.device, num_classes: int):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    true_indices: list[int] = []
    predicted_indices: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        true_indices.extend(labels.cpu().tolist())
        predicted_indices.extend(predictions.cpu().tolist())
    metrics, _, _ = confusion_metrics(true_indices, predicted_indices, num_classes)
    return loss_sum / sample_count, metrics


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_fixed_training_monitor(
    model_name: str,
    seed: int,
    num_classes: int,
    training_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    data_root: Path,
    class_to_idx: dict[str, int],
    train_transform,
    eval_transform,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
):
    set_seed(seed)
    model = build_model(model_name, num_classes, device, "head")
    training_loader = make_loader(
        training_rows,
        data_root,
        class_to_idx,
        train_transform,
        args.batch_size,
        args.num_workers,
        True,
        seed,
    )
    validation_loader = make_loader(
        validation_rows,
        data_root,
        class_to_idx,
        eval_transform,
        args.batch_size,
        args.num_workers,
        False,
        seed,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        trainable_head_parameters(model, model_name),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    history: list[dict] = []
    train_seconds_total = 0.0
    validation_seconds_total = 0.0
    selection_start = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        synchronize(device)
        start = time.perf_counter()
        training_loss = train_one_epoch(
            model, model_name, training_loader, criterion, optimizer, device
        )
        synchronize(device)
        train_seconds = time.perf_counter() - start
        train_seconds_total += train_seconds

        synchronize(device)
        start = time.perf_counter()
        validation_loss, validation_metrics = evaluate_loader(
            model, validation_loader, criterion, device, num_classes
        )
        synchronize(device)
        validation_seconds = time.perf_counter() - start
        validation_seconds_total += validation_seconds

        is_recorded_epoch = epoch == args.max_epochs
        if is_recorded_epoch:
            torch.save(
                {
                    "model": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_metrics": validation_metrics,
                    "head_state_dict": classification_head(model, model_name).state_dict(),
                },
                run_dir / "fixed_epoch10_selection_head.pt",
            )

        history.append(
            {
                "phase": "model_selection",
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "is_recorded_epoch": int(is_recorded_epoch),
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
            }
        )
        print(
            f"[{model_name} seed={seed}] selection epoch={epoch}/{args.max_epochs} "
            f"train_loss={training_loss:.5f} val_loss={validation_loss:.5f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} "
            f"lr={learning_rate:.2e} recorded={int(is_recorded_epoch)}",
            flush=True,
        )
        scheduler.step(validation_metrics["macro_f1"])

    synchronize(device)
    selection_seconds = time.perf_counter() - selection_start
    recorded_epoch = args.max_epochs
    recorded_history = history[-1]
    timing = {
        "selection_seconds": selection_seconds,
        "selection_training_seconds": train_seconds_total,
        "selection_validation_seconds": validation_seconds_total,
    }
    del training_loader, validation_loader, optimizer, scheduler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return recorded_epoch, recorded_history, history, timing


def retrain_on_full_support(
    model_name: str,
    seed: int,
    num_classes: int,
    support_rows: list[dict[str, str]],
    data_root: Path,
    class_to_idx: dict[str, int],
    train_transform,
    fixed_epochs: int,
    learning_rates: list[float],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
):
    set_seed(seed)
    model = build_model(model_name, num_classes, device, "head")
    loader = make_loader(
        support_rows,
        data_root,
        class_to_idx,
        train_transform,
        args.batch_size,
        args.num_workers,
        True,
        seed + 10_000_019,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        trainable_head_parameters(model, model_name),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history: list[dict] = []
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, fixed_epochs + 1):
        learning_rate = learning_rates[epoch - 1]
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        synchronize(device)
        epoch_start = time.perf_counter()
        training_loss = train_one_epoch(
            model, model_name, loader, criterion, optimizer, device
        )
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "phase": "full_support_retrain",
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": training_loss,
                "validation_loss": "",
                "validation_accuracy": "",
                "validation_macro_precision": "",
                "validation_macro_recall": "",
                "validation_macro_f1": "",
                "is_recorded_epoch": "",
                "train_seconds": epoch_seconds,
                "validation_seconds": "",
            }
        )
        print(
            f"[{model_name} seed={seed}] full-support epoch={epoch}/{fixed_epochs} "
            f"train_loss={training_loss:.5f} lr={learning_rate:.2e}",
            flush=True,
        )
    synchronize(device)
    training_seconds = time.perf_counter() - start
    torch.save(
        {
            "model": model_name,
            "seed": seed,
            "epochs": fixed_epochs,
            "head_state_dict": classification_head(model, model_name).state_dict(),
        },
        run_dir / "final_full_support_epoch10_head.pt",
    )
    del loader, optimizer
    return model, history, training_seconds


@torch.inference_mode()
def predict_fixed_evaluation(
    model,
    model_name: str,
    seed: int,
    evaluation_rows: list[dict[str, str]],
    data_root: Path,
    class_order: list[str],
    class_to_idx: dict[str, int],
    eval_transform,
    warmup_images: int,
    device: torch.device,
):
    model.eval()
    idx_to_class = {index: label for label, index in class_to_idx.items()}

    for row in evaluation_rows[: min(warmup_images, len(evaluation_rows))]:
        image = Image.open(data_root / row["path"]).convert("RGB")
        tensor = eval_transform(image).unsqueeze(0).to(device)
        model(tensor)
    synchronize(device)

    prediction_rows: list[dict] = []
    true_indices: list[int] = []
    predicted_indices: list[int] = []
    inference_seconds_total = 0.0
    forward_seconds_total = 0.0
    for index, row in enumerate(evaluation_rows, start=1):
        synchronize(device)
        end_to_end_start = time.perf_counter()
        image = Image.open(data_root / row["path"]).convert("RGB")
        tensor = eval_transform(image).unsqueeze(0).to(device)
        synchronize(device)
        forward_start = time.perf_counter()
        logits = model(tensor)
        synchronize(device)
        forward_seconds = time.perf_counter() - forward_start
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())
        end_to_end_seconds = time.perf_counter() - end_to_end_start

        true_index = class_to_idx[row["label"]]
        true_indices.append(true_index)
        predicted_indices.append(predicted_index)
        inference_seconds_total += end_to_end_seconds
        forward_seconds_total += forward_seconds
        prediction_rows.append(
            {
                "model": model_name,
                "model_display_name": MODEL_DISPLAY_NAMES[model_name],
                "seed": seed,
                "target_id": row["target_id"],
                "target_path": row["path"],
                "true_label": row["label"],
                "pred_label": idx_to_class[predicted_index],
                "correct": int(true_index == predicted_index),
                "confidence": confidence,
                "inference_seconds": end_to_end_seconds,
                "forward_seconds": forward_seconds,
            }
        )
        if index % 100 == 0 or index == len(evaluation_rows):
            print(
                f"[{model_name} seed={seed}] evaluated {index}/{len(evaluation_rows)} images",
                flush=True,
            )

    metrics, matrix, per_class = confusion_metrics(
        true_indices, predicted_indices, len(class_order)
    )
    timing = {
        "inference_seconds_total": inference_seconds_total,
        "average_inference_seconds_per_image": inference_seconds_total / len(evaluation_rows),
        "forward_seconds_total": forward_seconds_total,
        "average_forward_seconds_per_image": forward_seconds_total / len(evaluation_rows),
        "inference_batch_size": 1,
        "warmup_images": min(warmup_images, len(evaluation_rows)),
    }
    return prediction_rows, metrics, matrix, per_class, timing


def write_run_outputs(
    run_dir: Path,
    class_order: list[str],
    predictions: list[dict],
    history: list[dict],
    matrix: list[list[int]],
    per_class: list[dict],
    run_summary: dict,
) -> None:
    write_csv(
        run_dir / "predictions.csv",
        predictions,
        [
            "model",
            "model_display_name",
            "seed",
            "target_id",
            "target_path",
            "true_label",
            "pred_label",
            "correct",
            "confidence",
            "inference_seconds",
            "forward_seconds",
        ],
    )
    write_csv(
        run_dir / "epoch_history.csv",
        history,
        [
            "phase",
            "model",
            "seed",
            "epoch",
            "learning_rate",
            "train_loss",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_precision",
            "validation_macro_recall",
            "validation_macro_f1",
            "is_recorded_epoch",
            "train_seconds",
            "validation_seconds",
        ],
    )
    matrix_rows = []
    for label, values in zip(class_order, matrix):
        matrix_rows.append({"true_label": label, **dict(zip(class_order, values))})
    write_csv(
        run_dir / "confusion_matrix_counts.csv",
        matrix_rows,
        ["true_label", *class_order],
    )
    normalized_rows = []
    for label, values in zip(class_order, matrix):
        row_total = sum(values)
        normalized = [value / row_total if row_total else 0.0 for value in values]
        normalized_rows.append({"true_label": label, **dict(zip(class_order, normalized))})
    write_csv(
        run_dir / "confusion_matrix_normalized.csv",
        normalized_rows,
        ["true_label", *class_order],
    )
    per_class_rows = []
    for label, values in zip(class_order, per_class):
        per_class_rows.append({"class": label, **values})
    write_csv(
        run_dir / "per_class_metrics.csv",
        per_class_rows,
        ["class", "support", "precision", "recall", "f1"],
    )
    write_json(run_dir / "run_summary.json", run_summary)


def aggregate_runs(out_dir: Path, models: list[str], seeds: list[int]) -> None:
    run_summaries: list[dict] = []
    for model_name in models:
        for seed in seeds:
            path = out_dir / model_name / f"seed_{seed}" / "run_summary.json"
            if path.exists():
                run_summaries.append(read_json(path))

    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    timing_keys = [
        "train_validation_seconds_total",
        "average_inference_seconds_per_image",
        "average_forward_seconds_per_image",
    ]
    aggregate: dict[str, dict] = {}
    for model_name in models:
        rows = [row for row in run_summaries if row["model"] == model_name]
        if not rows:
            continue
        model_aggregate = {"completed_seeds": [row["seed"] for row in rows], "num_runs": len(rows)}
        for key in metric_keys + timing_keys:
            values = [float(row[key]) for row in rows]
            model_aggregate[f"{key}_mean"] = statistics.fmean(values)
            model_aggregate[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregate[model_name] = model_aggregate

    write_json(
        out_dir / "summary.json",
        {"runs": run_summaries, "aggregate_population_std": aggregate},
    )
    fields = [
        "dataset",
        "model",
        "model_display_name",
        "seed",
        "num_classes",
        "evaluation_size",
        "support_size",
        "selection_train_size",
        "selection_validation_size",
        "recorded_epoch",
        *metric_keys,
        "selection_seconds",
        "final_retrain_seconds",
        *timing_keys,
    ]
    metric_rows = [{field: row.get(field, "") for field in fields} for row in run_summaries]
    write_csv(out_dir / "metrics_by_seed.csv", metric_rows, fields)

    aggregate_rows = []
    for model_name, values in aggregate.items():
        aggregate_rows.append(
            {
                "model": model_name,
                "model_display_name": MODEL_DISPLAY_NAMES[model_name],
                **values,
            }
        )
    aggregate_fields = ["model", "model_display_name", "completed_seeds", "num_runs"]
    for key in metric_keys + timing_keys:
        aggregate_fields.extend([f"{key}_mean", f"{key}_std"])
    write_csv(out_dir / "metrics_mean_std.csv", aggregate_rows, aggregate_fields)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    manifest_dir = resolve_project_path(args.manifest_dir, project_root)
    out_dir = resolve_project_path(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_summary = read_json(manifest_dir / "summary.json")
    class_order = read_json(manifest_dir / "class_order.json")["classes"]
    support_rows = read_csv(manifest_dir / "support.csv")
    evaluation_rows = read_csv(manifest_dir / "evaluation.csv")
    support_rows = cap_per_class(support_rows, class_order, args.limit_per_class)
    evaluation_rows = cap_per_class(evaluation_rows, class_order, args.limit_per_class)
    assert_manifest_is_leakage_free(
        manifest_summary,
        support_rows,
        evaluation_rows,
        class_order,
        args.limit_per_class,
    )

    data_root = resolve_project_path(manifest_summary["data_root"], project_root)
    missing = [row["path"] for row in support_rows + evaluation_rows if not (data_root / row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first missing path: {missing[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = {label: index for index, label in enumerate(class_order)}
    train_transform, eval_transform = make_transforms()
    run_config = {
        "protocol": "fixed-evaluation full-data head-only transfer learning",
        "dataset": manifest_summary["dataset"],
        "manifest_dir": str(manifest_dir),
        "manifest_split_mode": manifest_summary["split_mode"],
        "evaluation_role": "final_test_only_never_used_for_selection",
        "evaluation_per_class": args.limit_per_class or 100,
        "evaluation_size": len(evaluation_rows),
        "support_size": len(support_rows),
        "support_internal_validation_ratio": args.validation_ratio,
        "final_retraining": "reinitialize head and train on 100% support for exactly 10 epochs",
        "checkpoint_selection": "none; only the epoch-10 model is evaluated",
        "models": args.models,
        "seeds": args.seeds,
        "pretraining": {
            "resnet18": "torchvision ResNet18_Weights.IMAGENET1K_V1",
            "resnet50": "torchvision ResNet50_Weights.IMAGENET1K_V2",
            "vit_tiny": "timm vit_tiny_patch16_224 pretrained",
            "vit_small": "timm vit_small_patch16_224 pretrained",
        },
        "train_mode": "frozen_backbone_linear_head",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "early_stopping": False,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "internal_validation_macro_f1",
            "factor": args.scheduler_factor,
            "patience": args.scheduler_patience,
            "min_lr": args.min_lr,
        },
        "augmentation": [
            "RandomResizedCrop(224, scale=(0.8, 1.0))",
            "RandomHorizontalFlip(0.5)",
            "RandomVerticalFlip(0.5)",
            "RandomQuarterTurn",
        ],
        "evaluation_transform": "Resize(256), CenterCrop(224), ImageNet normalization",
        "metrics": ["accuracy", "macro_precision", "macro_recall", "macro_f1"],
        "inference_timing": "batch_size=1, end-to-end image load+preprocess+H2D+forward",
        "manifest_hashes": {
            "support_csv_sha256": sha256_file(manifest_dir / "support.csv"),
            "evaluation_csv_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        },
        "runtime": runtime_metadata(device),
        "smoke_limit_per_class": args.limit_per_class,
    }
    write_json(out_dir / "run_config.json", run_config)
    print(
        f"Dataset={manifest_summary['dataset']} device={device} support={len(support_rows)} "
        f"fixed_evaluation={len(evaluation_rows)} classes={len(class_order)}",
        flush=True,
    )

    for model_name in args.models:
        for seed in args.seeds:
            run_dir = out_dir / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            completed_path = run_dir / "run_summary.json"
            if args.resume and completed_path.exists() and (run_dir / "predictions.csv").exists():
                print(f"[{model_name} seed={seed}] already complete; skipping.", flush=True)
                continue

            selection_train, selection_validation, split_counts = stratified_support_split(
                support_rows, class_order, args.validation_ratio, seed
            )
            write_json(
                run_dir / "internal_split.json",
                {
                    "seed": seed,
                    "counts": split_counts,
                    "selection_train_paths": [row["path"] for row in selection_train],
                    "selection_validation_paths": [row["path"] for row in selection_validation],
                },
            )
            print(
                f"[{model_name} seed={seed}] selection train={len(selection_train)} "
                f"validation={len(selection_validation)}",
                flush=True,
            )
            recorded_epoch, recorded_history, selection_history, selection_timing = run_fixed_training_monitor(
                model_name,
                seed,
                len(class_order),
                selection_train,
                selection_validation,
                data_root,
                class_to_idx,
                train_transform,
                eval_transform,
                args,
                device,
                run_dir,
            )
            learning_rates = [float(row["learning_rate"]) for row in selection_history]
            model, retrain_history, final_retrain_seconds = retrain_on_full_support(
                model_name,
                seed,
                len(class_order),
                support_rows,
                data_root,
                class_to_idx,
                train_transform,
                recorded_epoch,
                learning_rates,
                args,
                device,
                run_dir,
            )
            predictions, metrics, matrix, per_class, inference_timing = predict_fixed_evaluation(
                model,
                model_name,
                seed,
                evaluation_rows,
                data_root,
                class_order,
                class_to_idx,
                eval_transform,
                args.warmup_images,
                device,
            )
            train_validation_seconds_total = (
                selection_timing["selection_seconds"] + final_retrain_seconds
            )
            run_summary = {
                "dataset": manifest_summary["dataset"],
                "model": model_name,
                "model_display_name": MODEL_DISPLAY_NAMES[model_name],
                "seed": seed,
                "num_classes": len(class_order),
                "evaluation_size": len(evaluation_rows),
                "support_size": len(support_rows),
                "selection_train_size": len(selection_train),
                "selection_validation_size": len(selection_validation),
                "recorded_epoch": recorded_epoch,
                "epoch10_validation_loss": recorded_history["validation_loss"],
                "epoch10_validation_accuracy": recorded_history["validation_accuracy"],
                "epoch10_validation_macro_precision": recorded_history["validation_macro_precision"],
                "epoch10_validation_macro_recall": recorded_history["validation_macro_recall"],
                "epoch10_validation_macro_f1": recorded_history["validation_macro_f1"],
                **metrics,
                **selection_timing,
                "final_retrain_seconds": final_retrain_seconds,
                "train_validation_seconds_total": train_validation_seconds_total,
                **inference_timing,
            }
            write_run_outputs(
                run_dir,
                class_order,
                predictions,
                selection_history + retrain_history,
                matrix,
                per_class,
                run_summary,
            )
            print(
                f"[{model_name} seed={seed}] FINAL accuracy={metrics['accuracy']:.4f} "
                f"macro_precision={metrics['macro_precision']:.4f} "
                f"macro_recall={metrics['macro_recall']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"infer={inference_timing['average_inference_seconds_per_image']:.4f}s/img",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            aggregate_runs(out_dir, args.models, args.seeds)

    aggregate_runs(out_dir, args.models, args.seeds)
    print(f"Completed fixed-evaluation full-data experiment: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
