from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rs_visemds.category_texts import description_ensembles
from rs_visemds.selector import min_max_normalize, select_demonstrations


class SelectorTests(unittest.TestCase):
    def test_min_max_constant_is_zero(self):
        result = min_max_normalize(np.array([0.5, 0.5], dtype=np.float32))
        np.testing.assert_allclose(result, np.zeros(2), atol=1e-7)

    def test_class_balanced_pool_and_top_k(self):
        target = unit([1.0, 0.0])
        support = np.stack([
            unit([1.0, 0.0]),
            unit([0.8, 0.2]),
            unit([0.7, 0.3]),
            unit([0.2, 0.8]),
            unit([0.0, 1.0]),
            unit([0.4, 0.6]),
        ])
        labels = ["a", "a", "a", "b", "b", "b"]
        prototypes = np.stack([unit([1.0, 0.0]), unit([0.0, 1.0])])
        selected, candidates = select_demonstrations(
            target, support, labels, prototypes, ["a", "b"], r=2, k=2
        )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(selected), 2)
        self.assertGreaterEqual(selected[0].score, selected[1].score)
        self.assertTrue({row.label for row in candidates} == {"a", "b"})

    def test_description_count(self):
        texts = description_ensembles(
            "nwpu_fg_urban",
            ["dense_residential", "medium_residential", "sparse_residential",
             "mobile_home_park", "commercial_area", "industrial_area",
             "parking_lot", "railway_station"],
        )
        self.assertTrue(all(len(items) == 10 for items in texts.values()))

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            select_demonstrations(
                unit([1.0, 0.0]),
                np.stack([unit([1.0, 0.0]), unit([0.0, 1.0])]),
                ["a", "b"],
                np.stack([unit([1.0, 0.0]), unit([0.0, 1.0])]),
                ["a", "b"], r=1, k=1, weights=(1.0, 1.0, 1.0),
            )


def unit(values):
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array)


if __name__ == "__main__":
    unittest.main()

