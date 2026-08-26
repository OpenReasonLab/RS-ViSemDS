from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


BASE_WEIGHTS = (0.6, 0.2, 0.2)
EPSILON = 1e-12


@dataclass(frozen=True)
class AdaptiveWeightResult:
    """Query-specific weights and their complete Eq. (7)-(13) audit trail."""

    visual_temperature: float
    semantic_temperature: float
    visual_concentration: float
    semantic_concentration: float
    visual_semantic_disagreement: float
    adjustment: float
    base_visual_proportion: float
    visual_proportion: float
    alpha: float
    beta: float
    gamma: float
    visual_class_evidence: tuple[float, ...]
    semantic_class_evidence: tuple[float, ...]
    visual_class_distribution: tuple[float, ...]
    semantic_class_distribution: tuple[float, ...]

    @property
    def weights(self) -> tuple[float, float, float]:
        return self.alpha, self.beta, self.gamma

    def to_dict(self) -> dict:
        return asdict(self)


def compute_adaptive_weights(
    target_embedding: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: list[str],
    category_prototypes: np.ndarray,
    class_order: list[str],
    *,
    visual_temperature: float,
    semantic_temperature: float,
    base_weights: tuple[float, float, float] = BASE_WEIGHTS,
    excluded_support_index: int | None = None,
) -> AdaptiveWeightResult:
    """Implement the paper's query-adaptive weighting equations (7)-(13).

    The visual log evidence uses every support image in the class, not the
    class-balanced top-r candidate subset. ``excluded_support_index`` is used
    only for support-only calibration, where a held-out support query must be
    excluded from its own class reference set.
    """
    _validate_inputs(
        target_embedding,
        support_embeddings,
        support_labels,
        category_prototypes,
        class_order,
        visual_temperature,
        semantic_temperature,
        excluded_support_index,
    )
    alpha0, beta0, gamma0 = validate_base_weights(base_weights)
    visual_evidence = class_visual_log_evidence(
        target_embedding,
        support_embeddings,
        support_labels,
        class_order,
        visual_temperature,
        excluded_support_index=excluded_support_index,
    )
    semantic_evidence = (
        np.asarray(category_prototypes, dtype=np.float64)
        @ np.asarray(target_embedding, dtype=np.float64)
    ) / semantic_temperature
    p_visual = softmax_distribution(visual_evidence)
    p_semantic = softmax_distribution(semantic_evidence)
    kappa_visual = evidence_concentration(p_visual)
    kappa_semantic = evidence_concentration(p_semantic)
    disagreement = normalized_jensen_shannon(p_visual, p_semantic)
    adjustment = disagreement * (kappa_visual - kappa_semantic)

    pi0 = alpha0 / (alpha0 + gamma0)
    logit_pi0 = math.log(pi0 / (1.0 - pi0))
    pi_visual = sigmoid(logit_pi0 + adjustment)
    alpha = (1.0 - beta0) * pi_visual
    beta = beta0
    gamma = (1.0 - beta0) * (1.0 - pi_visual)
    weights = validate_adaptive_weights((alpha, beta, gamma), beta0=beta0)

    return AdaptiveWeightResult(
        visual_temperature=float(visual_temperature),
        semantic_temperature=float(semantic_temperature),
        visual_concentration=kappa_visual,
        semantic_concentration=kappa_semantic,
        visual_semantic_disagreement=disagreement,
        adjustment=float(adjustment),
        base_visual_proportion=float(pi0),
        visual_proportion=float(pi_visual),
        alpha=weights[0],
        beta=weights[1],
        gamma=weights[2],
        visual_class_evidence=tuple(float(value) for value in visual_evidence),
        semantic_class_evidence=tuple(float(value) for value in semantic_evidence),
        visual_class_distribution=tuple(float(value) for value in p_visual),
        semantic_class_distribution=tuple(float(value) for value in p_semantic),
    )


def class_visual_log_evidence(
    target_embedding: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: list[str],
    class_order: list[str],
    temperature: float,
    *,
    excluded_support_index: int | None = None,
) -> np.ndarray:
    """Stable class-wise log-mean-exp over the complete support set (Eq. 7)."""
    temperature = _positive_temperature(temperature, "visual_temperature")
    target = np.asarray(target_embedding, dtype=np.float64)
    supports = np.asarray(support_embeddings, dtype=np.float64)
    similarities = supports @ target
    output: list[float] = []
    for label in class_order:
        indices = np.asarray(
            [index for index, value in enumerate(support_labels) if value == label],
            dtype=np.int64,
        )
        if excluded_support_index is not None:
            indices = indices[indices != excluded_support_index]
        if indices.size == 0:
            raise ValueError(
                f"Class {label!r} has no visual reference after query exclusion"
            )
        scaled = similarities[indices] / temperature
        maximum = float(np.max(scaled))
        log_mean_exp = maximum + math.log(
            float(np.mean(np.exp(scaled - maximum), dtype=np.float64))
        )
        output.append(log_mean_exp)
    return np.asarray(output, dtype=np.float64)


def softmax_distribution(log_evidence: np.ndarray) -> np.ndarray:
    values = np.asarray(log_evidence, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("softmax_distribution expects at least two class values")
    if not np.all(np.isfinite(values)):
        raise ValueError("class log evidence contains non-finite values")
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials, dtype=np.float64))


def normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = _validate_distribution(probabilities)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.clip(entropy / math.log(probabilities.size), 0.0, 1.0))


def evidence_concentration(probabilities: np.ndarray) -> float:
    return float(np.clip(1.0 - normalized_entropy(probabilities), 0.0, 1.0))


def normalized_jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = _validate_distribution(left)
    right = _validate_distribution(right)
    if left.shape != right.shape:
        raise ValueError("visual and semantic distributions must have equal shape")
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * (
        float(np.sum(left * np.log(left / midpoint)))
        + float(np.sum(right * np.log(right / midpoint)))
    )
    return float(np.clip(divergence / math.log(2.0), 0.0, 1.0))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def validate_base_weights(
    weights: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(weights) != 3:
        raise ValueError("base_weights must contain alpha0, beta0, gamma0")
    alpha0, beta0, gamma0 = (float(value) for value in weights)
    if any(not math.isfinite(value) or value < 0.0 for value in (
        alpha0, beta0, gamma0
    )):
        raise ValueError("base weights must be finite and non-negative")
    if alpha0 <= 0.0 or gamma0 <= 0.0 or beta0 >= 1.0:
        raise ValueError("the paper requires alpha0>0, gamma0>0, and beta0<1")
    if not math.isclose(alpha0 + beta0 + gamma0, 1.0, abs_tol=1e-9):
        raise ValueError("base weights must sum to one")
    return alpha0, beta0, gamma0


def validate_adaptive_weights(
    weights: tuple[float, float, float], *, beta0: float
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"adaptive weights must be finite and in [0, 1], got {values}")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError(f"adaptive weights must sum to one, got {values}")
    if not math.isclose(values[1], beta0, abs_tol=1e-12):
        raise ValueError(f"candidate typicality must remain fixed at beta0={beta0}")
    return values


def _validate_distribution(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("probability distribution must contain at least two classes")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("probabilities must be finite and strictly positive")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-9):
        raise ValueError("probabilities must sum to one")
    return probabilities


def _positive_temperature(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_inputs(
    target_embedding: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: list[str],
    category_prototypes: np.ndarray,
    class_order: list[str],
    visual_temperature: float,
    semantic_temperature: float,
    excluded_support_index: int | None,
) -> None:
    if target_embedding.ndim != 1 or not np.all(np.isfinite(target_embedding)):
        raise ValueError("target_embedding must be a finite 1D vector")
    if support_embeddings.ndim != 2 or not np.all(np.isfinite(support_embeddings)):
        raise ValueError("support_embeddings must be a finite 2D matrix")
    if support_embeddings.shape[0] != len(support_labels):
        raise ValueError("support_embeddings/support_labels length mismatch")
    if support_embeddings.shape[1] != target_embedding.shape[0]:
        raise ValueError("target/support embedding dimensions differ")
    if category_prototypes.shape != (len(class_order), target_embedding.shape[0]):
        raise ValueError("category_prototypes shape does not match classes/embedding size")
    if len(class_order) < 2 or len(set(class_order)) != len(class_order):
        raise ValueError("class_order must contain at least two unique labels")
    if set(support_labels) != set(class_order):
        raise ValueError("support labels and class_order must contain the same classes")
    _positive_temperature(visual_temperature, "visual_temperature")
    _positive_temperature(semantic_temperature, "semantic_temperature")
    if excluded_support_index is not None and not (
        0 <= excluded_support_index < len(support_labels)
    ):
        raise ValueError("excluded_support_index is out of range")
