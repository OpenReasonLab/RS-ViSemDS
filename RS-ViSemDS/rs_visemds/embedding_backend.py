from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from strict_fewshot.utils import read_csv, read_json, repo_path, sha256_file

from .category_texts import category_text_sha256, description_ensembles


ENCODER_NAME = "RemoteCLIP-ViT-B-32"
CACHE_VERSION = 1


@dataclass(frozen=True)
class EmbeddingBundle:
    support_embeddings: np.ndarray
    target_embeddings: np.ndarray
    category_prototypes: np.ndarray
    support_rows: list[dict[str, str]]
    target_rows: list[dict[str, str]]
    class_order: list[str]
    data_root: Path
    metadata: dict


def load_manifest(manifest_dir: Path, project_root: Path):
    evaluation = read_csv(manifest_dir / "evaluation.csv")
    support = read_csv(manifest_dir / "support.csv")
    class_order = read_json(manifest_dir / "class_order.json")["classes"]
    summary = read_json(manifest_dir / "summary.json")
    data_root = repo_path(summary["data_root"], project_root)
    return evaluation, support, class_order, summary, data_root


def load_or_build_embeddings(
    dataset: str,
    manifest_dir: Path,
    project_root: Path,
    cache_dir: Path,
    remoteclip_cache: Path,
    remoteclip_checkpoint: Path | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    force: bool = False,
) -> EmbeddingBundle:
    evaluation, support, class_order, summary, data_root = load_manifest(
        manifest_dir, project_root
    )
    checkpoint = _resolve_checkpoint(remoteclip_cache, remoteclip_checkpoint)
    expected_meta = _expected_metadata(
        dataset, manifest_dir, class_order, summary, checkpoint
    )
    cache_root = cache_dir / summary["dataset"] / "remoteclip_vit_b_32"
    paths = {
        "support": cache_root / "support_embeddings.npz",
        "target": cache_root / "target_embeddings.npz",
        "category": cache_root / "category_prototypes.npz",
        "metadata": cache_root / "metadata.json",
    }

    if not force and all(path.exists() for path in paths.values()):
        actual_meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        mismatches = {
            key: (actual_meta.get(key), value)
            for key, value in expected_meta.items()
            if actual_meta.get(key) != value
        }
        if not mismatches:
            return EmbeddingBundle(
                support_embeddings=_load_array(paths["support"], "embeddings"),
                target_embeddings=_load_array(paths["target"], "embeddings"),
                category_prototypes=_load_array(paths["category"], "prototypes"),
                support_rows=support,
                target_rows=evaluation,
                class_order=class_order,
                data_root=data_root,
                metadata=actual_meta,
            )
        print(f"Embedding cache metadata changed; rebuilding: {sorted(mismatches)}")

    missing_images = [
        str(row["path"])
        for row in support + evaluation
        if not (data_root / row["path"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing dataset images, first entries: {missing_images[:5]}")

    descriptions = description_ensembles(dataset, class_order)
    support_embeddings, target_embeddings, prototypes = _encode_remoteclip(
        checkpoint=checkpoint,
        support_paths=[data_root / row["path"] for row in support],
        target_paths=[data_root / row["path"] for row in evaluation],
        description_groups=[descriptions[label] for label in class_order],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    _save_array(paths["support"], "embeddings", support_embeddings)
    _save_array(paths["target"], "embeddings", target_embeddings)
    _save_array(paths["category"], "prototypes", prototypes)
    paths["metadata"].write_text(
        json.dumps(expected_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return EmbeddingBundle(
        support_embeddings=support_embeddings,
        target_embeddings=target_embeddings,
        category_prototypes=prototypes,
        support_rows=support,
        target_rows=evaluation,
        class_order=class_order,
        data_root=data_root,
        metadata=expected_meta,
    )


def _expected_metadata(dataset, manifest_dir, class_order, summary, checkpoint) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "dataset": summary["dataset"],
        "dataset_argument": dataset,
        "encoder": ENCODER_NAME,
        "prototype_formula": "normalize(mean(normalize(each_description_embedding)))",
        "descriptions_per_class": 10,
        "l2_normalized_embeddings": True,
        "manifest_dir": str(manifest_dir.resolve()),
        "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        "support_sha256": sha256_file(manifest_dir / "support.csv"),
        "class_order_sha256": sha256_file(manifest_dir / "class_order.json"),
        "category_text_sha256": category_text_sha256(dataset, class_order),
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _resolve_checkpoint(cache_dir: Path, checkpoint: Path | None) -> Path:
    from strict_fewshot.features import resolve_remoteclip_checkpoint

    return resolve_remoteclip_checkpoint(cache_dir, checkpoint)


def _encode_remoteclip(
    checkpoint: Path,
    support_paths: list[Path],
    target_paths: list[Path],
    description_groups: list[list[str]],
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open_clip
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "RemoteCLIP checkpoint mismatch: "
            f"missing={incompatible.missing_keys[:5]}, "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    model = model.to(device).eval()
    support = _encode_images(
        model, preprocess, support_paths, device, batch_size, num_workers
    )
    targets = _encode_images(
        model, preprocess, target_paths, device, batch_size, num_workers
    )
    flat_texts = [text for group in description_groups for text in group]
    text_embeddings = _encode_texts(model, tokenizer, flat_texts, device, batch_size)
    prototypes = []
    start = 0
    for group in description_groups:
        group_embeddings = text_embeddings[start:start + len(group)]
        mean = group_embeddings.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm <= 0:
            raise ValueError("Zero-norm category prototype")
        prototypes.append(mean / norm)
        start += len(group)
    return support, targets, np.asarray(prototypes, dtype=np.float32)


def _encode_images(model, preprocess, paths, device, batch_size, num_workers):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class ImageDataset(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            with Image.open(paths[index]) as image:
                return preprocess(image.convert("RGB"))

    loader = DataLoader(
        ImageDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    output = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                features = model.encode_image(images)
            output.append(F.normalize(features.float(), p=2, dim=1).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _encode_texts(model, tokenizer, texts, device, batch_size):
    import torch
    import torch.nn.functional as F

    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start + batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                features = model.encode_text(tokens)
            output.append(F.normalize(features.float(), p=2, dim=1).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _save_array(path: Path, key: str, value: np.ndarray) -> None:
    np.savez_compressed(path, **{key: value.astype(np.float32)})


def _load_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data[key].astype(np.float32)

