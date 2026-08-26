from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

@dataclass(frozen=True)
class TemperatureCalibrationResult:
    visual_temperature: float
    semantic_temperature: float
    visual_nll: float
    semantic_nll: float
    seed: int
    samples_per_class: int
    num_queries: int
    query_support_indices: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_temperatures(
    support_embeddings: np.ndarray,
    support_labels: list[str],
    category_prototypes: np.ndarray,
    class_order: list[str],
    *,
    samples_per_class: int = 50,
    seed: int = 42,
    minimum_temperature: float = 1e-3,
    maximum_temperature: float = 1.0,
    iterations: int = 48,
) -> TemperatureCalibrationResult:
    """Support-only stratified temperature calibration.

    Temperatures are fitted separately by minimizing multiclass negative log
    likelihood. Every calibration query is a support image and is excluded from
    its own visual reference set, exactly as required by the paper. Candidate
    retrieval is not needed by the temperature objective; when used elsewhere,
    callers must likewise exclude the held-out query from that candidate pool.
    """
    embeddings = np.asarray(support_embeddings, dtype=np.float64)
    prototypes = np.asarray(category_prototypes, dtype=np.float64)
    _validate_calibration_inputs(
        embeddings,
        support_labels,
        prototypes,
        class_order,
        samples_per_class,
        minimum_temperature,
        maximum_temperature,
        iterations,
    )
    query_indices = stratified_calibration_indices(
        support_labels,
        class_order,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    label_to_index = {label: index for index, label in enumerate(class_order)}
    targets = np.asarray(
        [label_to_index[support_labels[index]] for index in query_indices],
        dtype=np.int64,
    )
    similarities = embeddings[query_indices] @ embeddings.T
    semantic_similarities = embeddings[query_indices] @ prototypes.T
    class_indices = [
        np.asarray(
            [index for index, value in enumerate(support_labels) if value == label],
            dtype=np.int64,
        )
        for label in class_order
    ]

    def visual_objective(temperature: float) -> float:
        evidence = np.empty((len(query_indices), len(class_order)), dtype=np.float64)
        for query_position, support_index in enumerate(query_indices):
            for class_position, indices in enumerate(class_indices):
                references = indices[indices != support_index]
                scaled = similarities[query_position, references] / temperature
                maximum = float(np.max(scaled))
                evidence[query_position, class_position] = maximum + math.log(
                    float(np.mean(np.exp(scaled - maximum), dtype=np.float64))
                )
        return multiclass_nll(evidence, targets)

    def semantic_objective(temperature: float) -> float:
        return multiclass_nll(semantic_similarities / temperature, targets)

    visual_temperature, visual_nll = minimize_positive_scalar(
        visual_objective,
        minimum_temperature,
        maximum_temperature,
        iterations=iterations,
    )
    semantic_temperature, semantic_nll = minimize_positive_scalar(
        semantic_objective,
        minimum_temperature,
        maximum_temperature,
        iterations=iterations,
    )
    return TemperatureCalibrationResult(
        visual_temperature=visual_temperature,
        semantic_temperature=semantic_temperature,
        visual_nll=visual_nll,
        semantic_nll=semantic_nll,
        seed=int(seed),
        samples_per_class=int(samples_per_class),
        num_queries=len(query_indices),
        query_support_indices=tuple(int(index) for index in query_indices),
    )


def stratified_calibration_indices(
    support_labels: list[str],
    class_order: list[str],
    *,
    samples_per_class: int,
    seed: int,
) -> list[int]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in class_order:
        indices = np.asarray(
            [index for index, value in enumerate(support_labels) if value == label],
            dtype=np.int64,
        )
        if indices.size < 2:
            raise ValueError(f"Class {label!r} needs at least two support images")
        count = min(samples_per_class, int(indices.size))
        chosen = rng.choice(indices, size=count, replace=False)
        selected.extend(int(index) for index in np.sort(chosen))
    return selected


def multiclass_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("logits/targets shape mismatch")
    maximum = np.max(logits, axis=1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(
        np.sum(np.exp(logits - maximum), axis=1, dtype=np.float64)
    )
    true_logits = logits[np.arange(logits.shape[0]), targets]
    return float(np.mean(log_normalizer - true_logits, dtype=np.float64))


def minimize_positive_scalar(
    objective,
    minimum: float,
    maximum: float,
    *,
    iterations: int = 48,
) -> tuple[float, float]:
    """Deterministic golden-section minimization in log-temperature space."""
    if not (0.0 < minimum < maximum) or iterations <= 0:
        raise ValueError("temperature bounds/iterations are invalid")
    left, right = math.log(minimum), math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = float(objective(math.exp(c)))
    fd = float(objective(math.exp(d)))
    for _ in range(iterations):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = float(objective(math.exp(c)))
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = float(objective(math.exp(d)))
    log_temperature = 0.5 * (left + right)
    temperature = math.exp(log_temperature)
    return float(temperature), float(objective(temperature))


def _validate_calibration_inputs(
    embeddings: np.ndarray,
    support_labels: list[str],
    prototypes: np.ndarray,
    class_order: list[str],
    samples_per_class: int,
    minimum_temperature: float,
    maximum_temperature: float,
    iterations: int,
) -> None:
    if embeddings.ndim != 2 or not np.all(np.isfinite(embeddings)):
        raise ValueError("support_embeddings must be a finite 2D matrix")
    if embeddings.shape[0] != len(support_labels):
        raise ValueError("support_embeddings/support_labels length mismatch")
    if prototypes.shape != (len(class_order), embeddings.shape[1]):
        raise ValueError("category_prototypes shape mismatch")
    if set(support_labels) != set(class_order):
        raise ValueError("support labels and class_order differ")
    if samples_per_class <= 0 or iterations <= 0:
        raise ValueError("calibration sample count and iterations must be positive")
    if not (0.0 < minimum_temperature < maximum_temperature):
        raise ValueError("temperature search bounds must satisfy 0 < min < max")
