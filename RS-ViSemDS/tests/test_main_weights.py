from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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


class MainWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_script("rs_builder_defaults", "build_rs_visemds_examples.py")
        cls.suite = load_script("rs_suite_defaults", "run_rs_visemds_all.py")

    def test_builder_defaults_match_paper(self) -> None:
        with patch.object(sys, "argv", ["builder", "--dataset", "aid", "--manifest-dir", "m", "--out-dir", "o"]):
            args = self.builder.parse_args()
        self.assertEqual((args.alpha, args.beta, args.gamma), (0.6, 0.2, 0.2))

    def test_main_suite_defaults_match_paper(self) -> None:
        with patch.object(sys, "argv", ["suite"]):
            args = self.suite.parse_args()
        self.assertEqual((args.alpha, args.beta, args.gamma), (0.6, 0.2, 0.2))
        self.assertEqual(args.prompt_mode, "manuscript_v1")

    def test_weight_tag_separates_uniform_ablation(self) -> None:
        paper = self.suite._weight_tag(0.6, 0.2, 0.2)
        uniform = self.suite._weight_tag(1 / 3, 1 / 3, 1 / 3)
        self.assertEqual(paper, "a0p6_b0p2_g0p2")
        self.assertNotEqual(paper, uniform)

    def test_uniform_selection_cannot_be_reused_as_paper_main(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest"
            manifest.mkdir()
            for name in ("evaluation.csv", "support.csv", "class_order.json"):
                (manifest / name).write_text(name, encoding="utf-8")
            selected = root / "examples_rs_visemds_shot_3.csv"
            selected.write_text("selected", encoding="utf-8")
            config_path = root / "selection_config.json"
            config = {
                "method": "RS-ViSemDS",
                "dataset": "aid",
                "r_per_class": 3,
                "k_total_demonstrations": 3,
                "weights": {"alpha": 1 / 3, "beta": 1 / 3, "gamma": 1 / 3},
                "evaluation_sha256": self.suite.sha256_file(manifest / "evaluation.csv"),
                "support_sha256": self.suite.sha256_file(manifest / "support.csv"),
                "class_order_sha256": self.suite.sha256_file(manifest / "class_order.json"),
                "selected_examples_sha256": self.suite.sha256_file(selected),
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            compatible = self.suite._selection_is_compatible(
                config_path, selected, manifest, "aid", 3, 3, (0.6, 0.2, 0.2)
            )
            self.assertFalse(compatible)


if __name__ == "__main__":
    unittest.main()
