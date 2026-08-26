from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rs_visemds.adaptive_weights import (
    BASE_WEIGHTS,
    class_visual_log_evidence,
    compute_adaptive_weights,
    evidence_concentration,
    normalized_jensen_shannon,
    softmax_distribution,
)
from rs_visemds.selector import class_balanced_candidates, select_adaptive_demonstrations


class AdaptiveWeightTests(unittest.TestCase):
    def setUp(self):
        self.class_order = ["a", "b", "c"]
        self.support_labels = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
        self.support = np.stack([
            unit([1.0, 0.0, 0.1]), unit([0.8, 0.1, 0.2]), unit([0.7, 0.2, 0.3]),
            unit([0.1, 1.0, 0.0]), unit([0.2, 0.8, 0.1]), unit([0.3, 0.7, 0.2]),
            unit([0.0, 0.1, 1.0]), unit([0.1, 0.2, 0.8]), unit([0.2, 0.3, 0.7]),
        ])
        self.prototypes = np.stack([
            unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0]), unit([0.0, 0.0, 1.0])
        ])
        self.target = unit([0.7, 0.2, 0.1])

    def compute(self):
        return compute_adaptive_weights(
            self.target,
            self.support,
            self.support_labels,
            self.prototypes,
            self.class_order,
            visual_temperature=0.2,
            semantic_temperature=0.1,
        )

    def test_eq7_uses_all_support_and_stable_log_mean_exp(self):
        evidence = class_visual_log_evidence(
            self.target,
            self.support,
            self.support_labels,
            self.class_order,
            0.2,
        )
        similarities = self.support[:3] @ self.target
        maximum = float(np.max(similarities / 0.2))
        expected = maximum + math.log(float(np.mean(np.exp(similarities / 0.2 - maximum))))
        self.assertAlmostEqual(float(evidence[0]), expected, places=12)

    def test_held_out_support_query_is_excluded_from_visual_reference(self):
        query = self.support[0]
        included = class_visual_log_evidence(
            query, self.support, self.support_labels, self.class_order, 0.2
        )
        excluded = class_visual_log_evidence(
            query,
            self.support,
            self.support_labels,
            self.class_order,
            0.2,
            excluded_support_index=0,
        )
        self.assertLess(excluded[0], included[0])

    def test_eq8_uses_softmax(self):
        scores = np.asarray([-0.5, 0.0, 0.5])
        actual = softmax_distribution(scores)
        expected = np.exp(scores - scores.max())
        expected /= expected.sum()
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_eq9_to_13_match_paper(self):
        result = self.compute()
        k_vis = evidence_concentration(np.asarray(result.visual_class_distribution))
        k_sem = evidence_concentration(np.asarray(result.semantic_class_distribution))
        d_vs = normalized_jensen_shannon(
            np.asarray(result.visual_class_distribution),
            np.asarray(result.semantic_class_distribution),
        )
        g = d_vs * (k_vis - k_sem)
        pi0 = BASE_WEIGHTS[0] / (BASE_WEIGHTS[0] + BASE_WEIGHTS[2])
        expected_pi = 1.0 / (1.0 + math.exp(-(math.log(pi0 / (1 - pi0)) + g)))
        self.assertAlmostEqual(result.adjustment, g, places=12)
        self.assertAlmostEqual(result.visual_proportion, expected_pi, places=12)
        self.assertAlmostEqual(result.alpha, 0.8 * expected_pi, places=12)
        self.assertAlmostEqual(result.beta, 0.2, places=12)
        self.assertAlmostEqual(result.gamma, 0.8 * (1 - expected_pi), places=12)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=12)

    def test_equal_distributions_retain_base_prior(self):
        target = unit([1.0, 1.0, 1.0])
        result = compute_adaptive_weights(
            target,
            np.stack([target, target, target, target, target, target]),
            ["a", "a", "b", "b", "c", "c"],
            np.stack([target, target, target]),
            self.class_order,
            visual_temperature=0.2,
            semantic_temperature=0.1,
        )
        np.testing.assert_allclose(result.weights, BASE_WEIGHTS, atol=1e-12)

    def test_candidate_pool_does_not_enter_weight_computation(self):
        parameters = set(inspect.signature(compute_adaptive_weights).parameters)
        self.assertNotIn("candidate_indices", parameters)
        self.assertNotIn("alpha_min", parameters)

    def test_adaptive_score_uses_all_three_components(self):
        weights = self.compute().weights
        candidates = class_balanced_candidates(
            self.target, self.support, self.support_labels, self.class_order, r=1
        )
        _, ranked = select_adaptive_demonstrations(
            self.target,
            self.support,
            self.support_labels,
            self.prototypes,
            self.class_order,
            candidates,
            weights,
            k=2,
        )
        for row in ranked:
            expected = (
                weights[0] * row.s_img_norm
                + weights[1] * row.s_typ_norm
                + weights[2] * row.s_sem_norm
            )
            self.assertAlmostEqual(row.score, expected, delta=1e-7)

    def test_target_label_cannot_enter_weighting_or_selector(self):
        forbidden = {"label", "ground_truth", "true_label", "target_label"}
        self.assertFalse(forbidden & set(inspect.signature(compute_adaptive_weights).parameters))
        self.assertFalse(
            forbidden & set(inspect.signature(select_adaptive_demonstrations).parameters)
        )

    def test_non_positive_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_adaptive_weights(
                self.target,
                self.support,
                self.support_labels,
                self.prototypes,
                self.class_order,
                visual_temperature=0.0,
                semantic_temperature=0.1,
            )


def unit(values):
    array = np.asarray(values, dtype=np.float64)
    return array / np.linalg.norm(array)


if __name__ == "__main__":
    unittest.main()
