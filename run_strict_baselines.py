from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from strict_fewshot.data import FewShotImageDataset, load_single_image
from strict_fewshot.metrics import summarize_predictions
from strict_fewshot.models import build_model, set_head_train_mode, trainable_head_parameters
from strict_fewshot.utils import (
    read_csv,
    read_json,
    repo_path,
    runtime_metadata,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--examples-csv", required=True)
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet50", "vit_tiny", "vit_small"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-mode", choices=["head"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


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


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    try:
        import numpy as np

        np.random.seed(worker_seed + worker_id)
    except ImportError:
        pass


def make_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def train_for_target(model, model_name, train_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    params = trainable_head_parameters(model, model_name)
    optimizer = torch.optim.Adam(params, lr=lr)

    set_head_train_mode(model, model_name)
    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_one(model, image_tensor):
    model.eval()
    logits = model(image_tensor)
    pred_idx = int(logits.argmax(dim=1).item())
    probs = torch.softmax(logits, dim=1)
    confidence = float(probs[0, pred_idx].item())
    return pred_idx, confidence


def main():
    args = parse_args()
    manifest_dir = repo_path(args.manifest_dir, Path.cwd())
    examples_csv = repo_path(args.examples_csv, Path.cwd())
    out_dir = repo_path(args.out_dir, Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(manifest_dir / "summary.json")
    class_order = read_json(manifest_dir / "class_order.json")["classes"]
    data_root = repo_path(summary["data_root"], Path.cwd())
    evaluation = read_csv(manifest_dir / "evaluation.csv")
    support = read_csv(manifest_dir / "support.csv")
    examples = read_csv(examples_csv)
    strategies = {row["strategy"] for row in examples}
    if strategies != {"knn"}:
        raise ValueError(
            "Traditional few-shot baselines require query-specific RemoteCLIP kNN examples. "
            f"Found strategies: {sorted(strategies)}"
        )
    retrieval_config_path = examples_csv.parent / "retrieval_config.json"
    if not retrieval_config_path.exists():
        raise FileNotFoundError(
            f"Missing kNN retrieval metadata: {retrieval_config_path}. "
            "Regenerate examples with build_examples.py --strategy knn."
        )
    retrieval_config = read_json(retrieval_config_path)
    if (
        retrieval_config.get("feature_backend") != "remoteclip"
        or retrieval_config.get("encoder") != "RemoteCLIP-ViT-B-32"
        or retrieval_config.get("similarity") != "cosine_similarity"
        or not retrieval_config.get("l2_normalized_embeddings")
    ):
        raise ValueError(
            "Traditional few-shot baselines require L2-normalized RemoteCLIP-ViT-B-32 "
            "embeddings with cosine similarity."
        )
    retrieval_config_hash = sha256_file(retrieval_config_path)
    examples_by_target = {}
    for row in examples:
        examples_by_target.setdefault(row["target_id"], []).append(row)

    evaluation_by_id = {row["target_id"]: row for row in evaluation}
    support_pairs = {(row["label"], row["path"]) for row in support}
    extra_targets = set(examples_by_target) - set(evaluation_by_id)
    missing_targets = set(evaluation_by_id) - set(examples_by_target)
    if extra_targets or missing_targets:
        raise ValueError(
            f"Example manifest target mismatch: {len(extra_targets)} extra, "
            f"{len(missing_targets)} missing."
        )

    class_to_idx = {cls: i for i, cls in enumerate(class_order)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    train_transform, eval_transform = make_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(out_dir / "run_config.json", {
        "dataset": summary["dataset"],
        "manifest_dir": args.manifest_dir,
        "examples_csv": args.examples_csv,
        "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        "support_sha256": sha256_file(manifest_dir / "support.csv"),
        "examples_sha256": sha256_file(examples_csv),
        "retrieval_config": retrieval_config,
        "retrieval_config_sha256": retrieval_config_hash,
        "models": args.models,
        "pretraining": {
            "resnet18": "torchvision ResNet18_Weights.IMAGENET1K_V1",
            "resnet50": "torchvision ResNet50_Weights.IMAGENET1K_V2",
            "vit_tiny": "timm vit_tiny_patch16_224 pretrained",
            "vit_small": "timm vit_small_patch16_224 pretrained",
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "loss": "cross_entropy",
        "train_mode": "head",
        "seed": args.seed,
        "runtime": runtime_metadata(device),
    })

    prediction_rows = []
    for model_index, model_name in enumerate(args.models):
        for target_index, target in enumerate(evaluation):
            target_id = target["target_id"]
            train_rows = examples_by_target.get(target_id)
            if not train_rows:
                raise ValueError(f"No examples found for target_id={target_id}")
            expected_shot = int(train_rows[0]["shot"])
            per_class_knn = retrieval_config.get("scope") == "per_class_support_pool"
            expected_train_size = expected_shot * len(class_order) if per_class_knn else expected_shot
            if len(train_rows) != expected_train_size:
                raise ValueError(
                    f"Target {target_id} has {len(train_rows)} examples, "
                    f"but shot={expected_shot} with scope={retrieval_config.get('scope')} "
                    f"requires exactly {expected_train_size} examples."
                )
            if len({row["example_path"] for row in train_rows}) != expected_train_size:
                raise ValueError(f"Target {target_id} contains duplicate examples.")
            ranked_rows = sorted(train_rows, key=lambda row: int(row["rank"]))
            expected_ranks = list(range(1, expected_train_size + 1))
            if [int(row["rank"]) for row in ranked_rows] != expected_ranks:
                raise ValueError(f"Target {target_id} has invalid kNN ranks.")
            if per_class_knn:
                labels = [row["example_label"] for row in train_rows]
                label_counts = {class_name: labels.count(class_name) for class_name in class_order}
                if any(count != expected_shot for count in label_counts.values()):
                    raise ValueError(
                        f"Target {target_id} is not class-balanced for {expected_shot}-shot: "
                        f"{label_counts}"
                    )
                for class_name in class_order:
                    class_scores = [
                        float(row["score"])
                        for row in ranked_rows
                        if row["example_label"] == class_name
                    ]
                    if any(left < right for left, right in zip(class_scores, class_scores[1:])):
                        raise ValueError(
                            f"Target {target_id} class={class_name} kNN scores are not descending."
                        )
            else:
                scores = [float(row["score"]) for row in ranked_rows]
                if any(left < right for left, right in zip(scores, scores[1:])):
                    raise ValueError(f"Target {target_id} kNN scores are not descending.")
            for row in train_rows:
                if row["target_label"] != target["label"] or row["target_path"] != target["path"]:
                    raise ValueError(f"Target metadata mismatch for target_id={target_id}")
                if (row["example_label"], row["example_path"]) not in support_pairs:
                    raise ValueError(
                        f"Example is not in the support pool: "
                        f"{row['example_label']} / {row['example_path']}"
                    )

            run_seed = args.seed + model_index * 1_000_003 + target_index * 10_007
            set_seed(run_seed)
            model = build_model(model_name, len(class_order), device, "head")
            generator = torch.Generator()
            generator.manual_seed(run_seed)
            train_dataset = FewShotImageDataset(train_rows, data_root, class_to_idx, train_transform)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker if args.num_workers > 0 else None,
                generator=generator,
            )

            total_start = time.perf_counter()
            train_start = time.perf_counter()
            train_for_target(model, model_name, train_loader, args.epochs, args.lr, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - train_start

            predict_start = time.perf_counter()
            image_tensor = load_single_image(data_root / target["path"], eval_transform, device)
            pred_idx, confidence = predict_one(model, image_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            predict_seconds = time.perf_counter() - predict_start
            total_seconds = time.perf_counter() - total_start
            pred_label = idx_to_class[pred_idx]
            correct = int(pred_label == target["label"])

            prediction_rows.append({
                "dataset": summary["dataset"],
                "strategy": train_rows[0]["strategy"],
                "shot": train_rows[0]["shot"],
                "model": model_name,
                "target_id": target_id,
                "target_path": target["path"],
                "true_label": target["label"],
                "pred_label": pred_label,
                "correct": correct,
                "confidence": f"{confidence:.8f}",
                "train_size": len(train_rows),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "train_mode": args.train_mode,
                "seed": args.seed,
                "sampling_seed": train_rows[0].get("sampling_seed", ""),
                "run_seed": run_seed,
                "train_seconds": f"{train_seconds:.4f}",
                "predict_seconds": f"{predict_seconds:.4f}",
                "total_seconds": f"{total_seconds:.4f}",
            })

            print(
                f"{model_name} {target_id}: true={target['label']} "
                f"pred={pred_label} correct={correct}"
            )

    pred_fields = [
        "dataset", "strategy", "shot", "model", "target_id", "target_path",
        "true_label", "pred_label", "correct", "confidence", "train_size",
        "epochs", "batch_size", "lr", "train_mode",
        "seed", "sampling_seed", "run_seed",
        "train_seconds", "predict_seconds", "total_seconds",
    ]
    write_csv(out_dir / "predictions.csv", prediction_rows, pred_fields)

    metrics, per_class_rows, matrices = summarize_predictions(prediction_rows, class_order)
    write_json(out_dir / "summary.json", metrics)
    write_csv(
        out_dir / "per_class_accuracy.csv",
        per_class_rows,
        ["model", "class", "support", "accuracy", "precision", "f1"],
    )
    for model_name, matrix in matrices.items():
        rows = []
        for true_label, values in zip(class_order, matrix):
            row = {"true_label": true_label}
            row.update({pred_label: value for pred_label, value in zip(class_order, values)})
            rows.append(row)
        write_csv(out_dir / f"confusion_matrix_{model_name}.csv", rows, ["true_label", *class_order])

    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
