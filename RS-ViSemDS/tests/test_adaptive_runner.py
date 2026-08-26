from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import run_rs_visemds as runner


class FakeModel:
    def __init__(self, output: str):
        self.output = output
        self.calls = 0

    def generate_from_messages(self, messages, images):
        self.calls += 1
        return self.output


class RaisingModel:
    def __init__(self):
        self.calls = 0

    def generate_from_messages(self, messages, images):
        self.calls += 1
        raise RuntimeError("synthetic failure")


class FailOnceModel:
    def __init__(self, output: str):
        self.output = output
        self.calls = 0

    def generate_from_messages(self, messages, images):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return self.output


class AdaptiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.class_order = ["a", "b"]
        self.support_labels = ["a", "a", "a", "b", "b", "b"]
        self.support_rows = [
            {"label": label, "path": f"support/{index}.jpg"}
            for index, label in enumerate(self.support_labels)
        ]
        self.support = np.stack(
            [
                unit([1.0, 0.0]),
                unit([0.9, 0.1]),
                unit([0.8, 0.2]),
                unit([0.0, 1.0]),
                unit([0.1, 0.9]),
                unit([0.2, 0.8]),
            ]
        )
        self.prototypes = np.stack([unit([1.0, 0.0]), unit([0.0, 1.0])])

    def test_cli_defaults_and_locked_contract(self):
        args = runner.parse_args(
            [
                "--dataset",
                "aid",
                "--weight-mode",
                "adaptive",
                "--output-dir",
                "outputs/test",
            ]
        )
        self.assertEqual(args.model_id, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(args.candidates_per_class, 3)
        self.assertEqual(args.num_demonstrations, 3)
        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.calibration_seed, 42)
        self.assertEqual(args.calibration_samples_per_class, 50)
        for key in (
            "visual_temperature",
            "semantic_temperature",
            "torch_dtype",
            "device_map",
            "max_tokens",
            "min_pixels",
            "max_pixels",
        ):
            self.assertIn(key, runner.RESUME_KEYS)

    def test_target_id_subset_requires_explicit_diagnostic_mode(self):
        with self.assertRaises(ValueError):
            runner.parse_args(
                [
                    "--dataset",
                    "aid",
                    "--weight-mode",
                    "adaptive",
                    "--output-dir",
                    "outputs/test",
                    "--target-ids-file",
                    "failed_ids.txt",
                ]
            )

    def test_diagnostic_target_filter_preserves_requested_order(self):
        evaluation = [
            {"target_id": "a", "label": "x", "path": "a.jpg"},
            {"target_id": "b", "label": "y", "path": "b.jpg"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("b\na\n", encoding="utf-8")
            filtered = runner.filter_evaluation_by_target_ids(evaluation, path)
        self.assertEqual([row["target_id"] for row in filtered], ["b", "a"])

    def test_one_target_uses_exactly_one_mllm_call_and_no_label_input(self):
        model = FakeModel('{"thoughts":"x","answer":"a","score":0.9}')
        parsed = {
            "pred_label": "a",
            "parse_valid": 1,
            "parse_mode": "json",
            "raw_pred_label": "a",
            "score": 0.9,
            "thoughts": "x",
        }
        with patch.object(
            runner, "build_local_messages_and_images", return_value=([{"role": "user"}], [])
        ), patch.object(runner, "parse_prediction", return_value=parsed):
            record = self._process(model)
        self.assertEqual(model.calls, 1)
        self.assertEqual(record["mllm_call_count"], 1)
        self.assertNotIn("ground_truth", record["selected_demonstrations"][0])
        self.assertIsNone(record["ground_truth"])
        self.assertFalse(record["invalid_output"])
        for required in (
            "image_id",
            "prediction",
            "visual_concentration",
            "semantic_concentration",
            "visual_semantic_disagreement",
            "adjustment",
            "alpha",
            "beta",
            "gamma",
            "selected_demonstrations",
            "raw_model_output",
            "total_time_seconds",
        ):
            self.assertIn(required, record)

    def test_invalid_output_is_recorded_and_counted_as_wrong(self):
        model = FakeModel("not valid json")
        parsed = {
            "pred_label": runner.INVALID_LABEL,
            "parse_valid": 0,
            "parse_mode": "invalid",
            "raw_pred_label": "",
            "score": "",
            "thoughts": "",
        }
        with patch.object(
            runner, "build_local_messages_and_images", return_value=([{"role": "user"}], [])
        ), patch.object(runner, "parse_prediction", return_value=parsed):
            record = self._process(model)
        record["ground_truth"] = "a"
        record["is_correct"] = False
        metrics, _, _ = runner.compute_summary(
            [record], self.class_order, 1, 0, 42, 0.0, 0.0
        )
        self.assertEqual(metrics["invalid_output_count"], 1)
        self.assertEqual(metrics["invalid_output_rate"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(model.calls, 1)

    def test_diagnostic_summary_is_never_marked_formal(self):
        record = runner.empty_prediction_record("x", "x.jpg")
        record.update(
            {
                "ground_truth": "a",
                "prediction": "a",
                "is_correct": True,
                "invalid_output": False,
                "mllm_call_count": 1,
            }
        )
        metrics, _, _ = runner.compute_summary(
            [record],
            self.class_order,
            1,
            0,
            42,
            0.0,
            0.0,
            formal_result_eligible=False,
        )
        self.assertTrue(metrics["run_complete"])
        self.assertFalse(metrics["formal_result_eligible"])
        self.assertEqual(
            metrics["result_status"], "DIAGNOSTIC_COMPLETE_NOT_FOR_PAPER"
        )

    def test_resume_keeps_invalid_complete_but_migrates_run_error_to_retry(self):
        rows = [
            {"image_id": "x", "invalid_output": True, "error": ""},
            {"image_id": "y", "invalid_output": False, "error": "RuntimeError: z"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            runner._write_jsonl_atomic(path, rows)
            loaded = runner.load_prediction_records(path)
        completed, errors = runner.migrate_legacy_error_rows(loaded)
        self.assertEqual({row["image_id"] for row in completed}, {"x"})
        self.assertEqual({row["image_id"] for row in errors}, {"y"})
        self.assertTrue(runner.can_attempt_target("y", errors, 3))

    def test_run_error_retry_limit_counts_retries_after_initial_attempt(self):
        errors = []
        self.assertTrue(runner.can_attempt_target("x", errors, 0))
        errors.append({"image_id": "x", "attempt": 1})
        self.assertFalse(runner.can_attempt_target("x", errors, 0))
        self.assertTrue(runner.can_attempt_target("x", errors, 3))
        errors.extend(
            {"image_id": "x", "attempt": attempt} for attempt in (2, 3, 4)
        )
        self.assertFalse(runner.can_attempt_target("x", errors, 3))

    def test_runtime_error_is_separate_from_invalid_output(self):
        model = RaisingModel()
        with patch.object(
            runner, "build_local_messages_and_images", return_value=([{"role": "user"}], [])
        ):
            record = self._process(model)
        self.assertEqual(model.calls, 1)
        self.assertIn("synthetic failure", record["error"])
        self.assertEqual(record["error_stage"], "qwen_inference")
        self.assertFalse(record["invalid_output"])
        self.assertEqual(record["mllm_call_count"], 1)

    def test_run_error_is_retried_on_resume_then_removed_from_unresolved_count(self):
        model = FailOnceModel('{"thoughts":"x","answer":"a","score":0.9}')
        parsed = {
            "pred_label": "a",
            "parse_valid": 1,
            "parse_mode": "json",
            "raw_pred_label": "a",
            "score": 0.9,
            "thoughts": "x",
        }
        completed = []
        errors = []
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, "build_local_messages_and_images", return_value=([{"role": "user"}], [])
        ), patch.object(runner, "parse_prediction", return_value=parsed):
            output = Path(directory)
            first = self._process(model)
            self.assertTrue(first["error"])
            errors.append(runner.build_run_error_event(first, attempt=1))
            runner._write_checkpoint_outputs(output, completed, errors)
            self.assertTrue((output / "run_errors.jsonl").read_text(encoding="utf-8").strip())
            self.assertNotIn("target-1", {row["image_id"] for row in completed})
            self.assertTrue(runner.can_attempt_target("target-1", errors, 3))

            second = self._process(model)
            self.assertFalse(second["error"])
            second["ground_truth"] = "a"
            second["is_correct"] = True
            completed.append(second)
            runner._write_checkpoint_outputs(output, completed, errors)
            unresolved = runner.unresolved_run_error_ids(
                completed, errors, ["target-1"]
            )
            self.assertEqual(unresolved, set())
            self.assertEqual(model.calls, 2)
            # A successful prediction is complete and must not be called again.
            self.assertIn("target-1", {row["image_id"] for row in completed})

    def test_all_required_output_files_are_written(self):
        record = runner.empty_prediction_record("x", "x.jpg")
        record.update(
            {
                "ground_truth": "a",
                "prediction": "a",
                "is_correct": True,
                "parse_valid": True,
                "alpha": 0.6,
                "beta": 0.2,
                "gamma": 0.2,
                "alpha_was_clamped": False,
                "visual_class_evidence": [0.5, 0.2],
                "semantic_class_evidence": [0.6, 0.1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner._write_all_outputs(
                out_dir=output,
                records=[record],
                class_order=self.class_order,
                expected_targets=1,
                bootstrap_samples=0,
                bootstrap_seed=42,
                embedding_preparation_seconds=0.1,
                invocation_wall_seconds=0.2,
                dataset_name="AID",
                model_id="Qwen/Qwen3-VL-8B-Instruct",
                run_errors=[],
                expected_target_ids=["x"],
            )
            runner.write_environment_file(
                output / "environment.txt", {"python_version": "test"}
            )
            expected = {
                "environment.txt",
                "predictions.jsonl",
                "invalid_outputs.jsonl",
                "run_errors.jsonl",
                "summary.json",
                "adaptive_weight_statistics.json",
                "per_class_metrics.csv",
                "confusion_matrix.csv",
                "confusion_matrix.png",
                "final_results.csv",
                "class_distribution_statistics.json",
                "selected_demonstration_statistics.json",
            }
            self.assertTrue(expected <= {path.name for path in output.iterdir()})

    def test_leakage_is_rejected(self):
        evaluation = [{"target_id": "t1", "path": "same.jpg", "label": "a"}]
        support = [{"support_id": "s1", "path": "same.jpg", "label": "a"}]
        with self.assertRaises(ValueError):
            runner.assert_no_target_leakage(evaluation, support)

    def _process(self, model):
        return runner.process_unlabeled_target(
            image_id="target-1",
            target_path="target.jpg",
            target_embedding=unit([0.9, 0.1]),
            dataset="aid",
            data_root=Path("."),
            class_order=self.class_order,
            support_rows=self.support_rows,
            support_embeddings=self.support,
            support_labels=self.support_labels,
            category_prototypes=self.prototypes,
            model=model,
            visual_temperature=0.2,
            semantic_temperature=0.1,
            candidates_per_class=3,
            num_demonstrations=3,
        )


def unit(values):
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array)


if __name__ == "__main__":
    unittest.main()
