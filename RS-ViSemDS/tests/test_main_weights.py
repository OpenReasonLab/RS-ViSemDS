from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


RS_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RS_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MainSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_script("rs_suite_defaults", "run_rs_visemds_all.py")

    def test_main_suite_uses_paper_models_datasets_and_ten_runs(self) -> None:
        with patch.object(sys, "argv", ["suite"]):
            args = self.suite.parse_args()
        self.assertEqual(args.datasets, ["aid", "nwpu_urban"])
        self.assertEqual(
            args.models, ["gemma3_12b", "qwen3vl_8b", "internvl35_14b"]
        )
        self.assertEqual(args.runs, 10)

    def test_main_suite_has_no_fixed_weight_or_legacy_prompt_arguments(self) -> None:
        with patch.object(sys, "argv", ["suite"]):
            args = self.suite.parse_args()
        self.assertFalse(hasattr(args, "alpha"))
        self.assertFalse(hasattr(args, "alpha_min"))
        self.assertFalse(hasattr(args, "prompt_mode"))

    def test_non_ten_run_formal_suite_is_rejected(self) -> None:
        with patch.object(sys, "argv", ["suite", "--runs", "3"]):
            with self.assertRaises(ValueError):
                self.suite.main()


if __name__ == "__main__":
    unittest.main()
