from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rs_visemds.calibration import (
    calibrate_temperatures,
    stratified_calibration_indices,
)


class CalibrationTests(unittest.TestCase):
    def test_stratified_indices_are_deterministic_and_balanced(self):
        labels = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
        left = stratified_calibration_indices(
            labels, ["a", "b", "c"], samples_per_class=3, seed=42
        )
        right = stratified_calibration_indices(
            labels, ["a", "b", "c"], samples_per_class=3, seed=42
        )
        self.assertEqual(left, right)
        self.assertEqual([sum(labels[index] == label for index in left) for label in "abc"], [3, 3, 3])

    def test_calibration_is_support_only_and_returns_positive_temperatures(self):
        embeddings = np.stack([
            unit([1.0, 0.0]), unit([0.9, 0.1]), unit([0.8, 0.2]),
            unit([0.0, 1.0]), unit([0.1, 0.9]), unit([0.2, 0.8]),
        ])
        result = calibrate_temperatures(
            embeddings,
            ["a", "a", "a", "b", "b", "b"],
            np.stack([unit([1.0, 0.0]), unit([0.0, 1.0])]),
            ["a", "b"],
            samples_per_class=2,
            iterations=8,
        )
        self.assertGreater(result.visual_temperature, 0.0)
        self.assertGreater(result.semantic_temperature, 0.0)
        self.assertEqual(result.num_queries, 4)
        self.assertEqual(len(set(result.query_support_indices)), 4)


def unit(values):
    array = np.asarray(values, dtype=np.float64)
    return array / np.linalg.norm(array)


if __name__ == "__main__":
    unittest.main()
