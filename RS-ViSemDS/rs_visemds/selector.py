from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ScoredCandidate:
    support_index: int
    label: str
    s_img: float
    s_typ: float
    s_sem: float
    s_img_norm: float
    s_typ_norm: float
    s_sem_norm: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def min_max_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("min_max_normalize expects a non-empty 1D array")
    return (values - values.min()) / (values.max() - values.min() + eps)


def class_balanced_candidates(
    target_embedding: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: list[str],
    class_order: list[str],
    r: int,
) -> list[int]:
    _validate_inputs(target_embedding, support_embeddings, support_labels)
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    visual_scores = support_embeddings @ target_embedding
    output: list[int] = []
    for label in class_order:
        indices = np.array(
            [index for index, value in enumerate(support_labels) if value == label],
            dtype=np.int64,
        )
        if len(indices) < r:
            raise ValueError(f"Class {label!r} has {len(indices)} support images; r={r}")
        local_order = np.argsort(-visual_scores[indices], kind="mergesort")[:r]
        output.extend(int(indices[index]) for index in local_order)
    return output


def score_candidates(
    target_embedding: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: list[str],
    category_prototypes: np.ndarray,
    class_order: list[str],
    candidate_indices: list[int],
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    eps: float = 1e-8,
) -> list[ScoredCandidate]:
    _validate_inputs(target_embedding, support_embeddings, support_labels)
    if not candidate_indices:
        raise ValueError("candidate_indices must not be empty")
    if category_prototypes.shape != (len(class_order), support_embeddings.shape[1]):
        raise ValueError(
            "category_prototypes must have shape "
            f"({len(class_order)}, {support_embeddings.shape[1]})"
        )
    alpha, beta, gamma = _validate_weights(weights)
    label_to_index = {label: index for index, label in enumerate(class_order)}
    labels = [support_labels[index] for index in candidate_indices]
    missing = sorted(set(labels) - set(label_to_index))
    if missing:
        raise ValueError(f"Candidate labels missing from class_order: {missing}")

    candidate_embeddings = support_embeddings[candidate_indices]
    text_embeddings = category_prototypes[
        np.array([label_to_index[label] for label in labels], dtype=np.int64)
    ]
    s_img = candidate_embeddings @ target_embedding
    s_typ = np.sum(candidate_embeddings * text_embeddings, axis=1)
    s_sem = text_embeddings @ target_embedding
    s_img_norm = min_max_normalize(s_img, eps)
    s_typ_norm = min_max_normalize(s_typ, eps)
    s_sem_norm = min_max_normalize(s_sem, eps)
    final = alpha * s_img_norm + beta * s_typ_norm + gamma * s_sem_norm

    return [
        ScoredCandidate(
            support_index=int(support_index),
            label=labels[local_index],
            s_img=float(s_img[local_index]),
            s_typ=float(s_typ[local_index]),
            s_sem=float(s_sem[local_index]),
            s_img_norm=float(s_img_norm[local_index]),
            s_typ_norm=float(s_typ_norm[local_index]),
            s_sem_norm=float(s_sem_norm[local_index]),
            score=float(final[local_index]),
        )
        for local_index, support_index in enumerate(candidate_indices)
    ]


def select_demonstrations(
    target_embedding: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: list[str],
    category_prototypes: np.ndarray,
    class_order: list[str],
    r: int = 3,
    k: int = 3,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    eps: float = 1e-8,
) -> tuple[list[ScoredCandidate], list[ScoredCandidate]]:
    """Return final top-k demonstrations and the full r*C candidate audit list."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    candidate_indices = class_balanced_candidates(
        target_embedding, support_embeddings, support_labels, class_order, r
    )
    if k > len(candidate_indices):
        raise ValueError(f"k={k} exceeds candidate pool size {len(candidate_indices)}")
    candidates = score_candidates(
        target_embedding,
        support_embeddings,
        support_labels,
        category_prototypes,
        class_order,
        candidate_indices,
        weights,
        eps,
    )
    # Stable, explicit tie-breaking keeps reruns identical across platforms.
    ranked = sorted(
        candidates,
        key=lambda row: (-row.score, -row.s_img, row.support_index),
    )
    return ranked[:k], ranked


def _validate_inputs(
    target_embedding: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: list[str],
) -> None:
    if target_embedding.ndim != 1:
        raise ValueError("target_embedding must be 1D")
    if support_embeddings.ndim != 2:
        raise ValueError("support_embeddings must be 2D")
    if support_embeddings.shape[0] != len(support_labels):
        raise ValueError("support_embeddings/support_labels length mismatch")
    if support_embeddings.shape[1] != target_embedding.shape[0]:
        raise ValueError("target/support embedding dimensions differ")


def _validate_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(weights) != 3:
        raise ValueError("weights must contain alpha, beta, gamma")
    values = tuple(float(value) for value in weights)
    if any(value < 0 for value in values):
        raise ValueError(f"weights must be non-negative, got {values}")
    if not np.isclose(sum(values), 1.0, atol=1e-7):
        raise ValueError(f"weights must sum to 1, got {values}")
    return values

