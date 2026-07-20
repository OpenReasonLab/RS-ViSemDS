from __future__ import annotations

import unittest

from run_zero_shot_mllm import summarize_rows
from strict_fewshot.metrics import summarize_predictions


class MacroMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.class_order = ["A", "B"]
        self.rows = [
            {"model": "m", "true_label": "A", "pred_label": "A", "correct": 1},
            {"model": "m", "true_label": "A", "pred_label": "A", "correct": 1},
            {"model": "m", "true_label": "B", "pred_label": "A", "correct": 0},
            {"model": "m", "true_label": "B", "pred_label": "B", "correct": 1},
        ]

    def assert_expected_metrics(self, metrics: dict) -> None:
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_precision"], (2 / 3 + 1) / 2)
        self.assertAlmostEqual(metrics["macro_recall"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 2 / 3) / 2)

    def test_mllm_summary_has_four_metrics(self) -> None:
        metrics, _, _ = summarize_rows(self.rows, self.class_order)
        self.assert_expected_metrics(metrics["m"])

    def test_traditional_summary_has_four_metrics(self) -> None:
        rows = [
            {
                **row,
                "train_seconds": 0.0,
                "predict_seconds": 0.0,
                "total_seconds": 0.0,
            }
            for row in self.rows
        ]
        metrics, _, _ = summarize_predictions(rows, self.class_order)
        self.assert_expected_metrics(metrics["m"])


if __name__ == "__main__":
    unittest.main()
