from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval100_protocol import validate_eval100_manifest
from strict_fewshot.models import build_model
from strict_fewshot.utils import (
    read_csv,
    read_json,
    repo_path,
    runtime_metadata,
    sha256_file,
    write_csv,
    write_json,
)


MODELS = ("resnet18", "resnet50", "vit_tiny", "vit_small")
PRETRAINING = {
    "resnet18": "torchvision ResNet18_Weights.IMAGENET1K_V1",
    "resnet50": "torchvision ResNet50_Weights.IMAGENET1K_V2",
    "vit_tiny": "timm vit_tiny_patch16_224 pretrained",
    "vit_small": "timm vit_small_patch16_224 pretrained",
}
FIELDS = (
    "target_id",
    "target_label",
    "target_path",
    "strategy",
    "shot",
    "rank",
    "example_label",
    "example_path",
    "score",
    "sampling_seed",
)


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], data_root: Path, transform) -> None:
        self.rows = rows
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path = self.data_root / self.rows[index]["path"]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build class-balanced, query-specific kNN selections with the same frozen "
            "ImageNet backbone used by each conventional visual baseline."
        )
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def build_feature_extractor(name: str, device: torch.device):
    model = build_model(name, num_classes=1, device=device, train_mode="head")
    if name in {"resnet18", "resnet50"}:
        model.fc = nn.Identity()
    else:
        model.head = nn.Identity()
    model.eval()
    return model


@torch.inference_mode()
def encode_rows(
    model,
    rows: list[dict[str, str]],
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        ManifestImageDataset(rows, data_root, make_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    for images in loader:
        features = model(images.to(device, non_blocking=True))
        chunks.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def stable_top_indices(scores: torch.Tensor, count: int) -> list[int]:
    values = scores.tolist()
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def main() -> None:
    args = parse_args()
    if not args.shots or any(shot <= 0 for shot in args.shots):
        raise ValueError("All shot values must be positive.")

    root = Path(__file__).resolve().parent
    manifest_dir = repo_path(args.manifest_dir, root)
    out_dir = repo_path(args.out_dir, root)
    validate_eval100_manifest(manifest_dir)

    summary = read_json(manifest_dir / "summary.json")
    class_order = read_json(manifest_dir / "class_order.json")["classes"]
    evaluation = read_csv(manifest_dir / "evaluation.csv")
    support = read_csv(manifest_dir / "support.csv")
    data_root = repo_path(summary["data_root"], root)
    missing = [row["path"] for row in evaluation + support if not (data_root / row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first missing path: {missing[0]}")

    support_by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(support):
        support_by_class[row["label"]].append(index)
    max_shot = max(args.shots)
    too_small = {
        label: len(support_by_class[label])
        for label in class_order
        if len(support_by_class[label]) < max_shot
    }
    if too_small:
        raise ValueError(f"Support classes smaller than max shot={max_shot}: {too_small}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor(args.model, device)
    support_features = encode_rows(
        model, support, data_root, args.batch_size, args.num_workers, device
    )
    evaluation_features = encode_rows(
        model, evaluation, data_root, args.batch_size, args.num_workers, device
    )

    selected_by_target: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for target_index, target in enumerate(evaluation):
        per_class: dict[str, list[tuple[int, float]]] = {}
        for label in class_order:
            support_indices = support_by_class[label]
            scores = support_features[support_indices] @ evaluation_features[target_index]
            local_indices = stable_top_indices(scores, max_shot)
            per_class[label] = [
                (support_indices[local_index], float(scores[local_index]))
                for local_index in local_indices
            ]
        selected_by_target[target["target_id"]] = per_class

    out_dir.mkdir(parents=True, exist_ok=True)
    for shot in sorted(set(args.shots)):
        rows = []
        for target in evaluation:
            rank = 0
            for label in class_order:
                for support_index, score in selected_by_target[target["target_id"]][label][:shot]:
                    rank += 1
                    example = support[support_index]
                    rows.append(
                        {
                            "target_id": target["target_id"],
                            "target_label": target["label"],
                            "target_path": target["path"],
                            "strategy": "knn",
                            "shot": shot,
                            "rank": rank,
                            "example_label": example["label"],
                            "example_path": example["path"],
                            "score": f"{score:.10f}",
                            "sampling_seed": "",
                        }
                    )
        write_csv(out_dir / f"examples_knn_shot_{shot}.csv", rows, list(FIELDS))

    write_json(
        out_dir / "retrieval_config.json",
        {
            "strategy": "knn",
            "feature_backend": "imagenet_backbone",
            "encoder": args.model,
            "pretraining": PRETRAINING[args.model],
            "scope": "per_class_support_pool",
            "shot_definition": "examples_per_class",
            "similarity": "cosine_similarity",
            "l2_normalized_embeddings": True,
            "nested_top_k": True,
            "input_size": "224x224",
            "seed": args.seed,
            "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
            "support_sha256": sha256_file(manifest_dir / "support.csv"),
            "runtime": runtime_metadata(device),
        },
    )
    print(f"Wrote {args.model} per-class backbone kNN selections to {out_dir}")


if __name__ == "__main__":
    main()
