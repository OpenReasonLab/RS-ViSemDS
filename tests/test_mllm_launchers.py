from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "run_gpt4o_aid_nwpu_all.py",
    "run_llama32_11b_aid_nwpu_all.py",
    "run_gemma3_12b_aid_nwpu_all.py",
    "run_qwen25vl_7b_aid_nwpu_all.py",
    "run_qwen3vl_8b_aid_nwpu_all.py",
    "run_internvl35_8b_aid_nwpu_all.py",
    "run_internvl35_14b_aid_nwpu_all.py",
)


class MLLMLauncherDryRunTests(unittest.TestCase):
    def test_clean_package_can_dry_run_all_three_settings(self) -> None:
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / launcher),
                        "--datasets",
                        "aid",
                        "--shots",
                        "3",
                        "--dry-run",
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("run_zero_shot_mllm.py", completed.stdout)
                self.assertIn("run_random_fewshot_mllm.py", completed.stdout)
                self.assertIn("run_knn_totalshot_mllm.py", completed.stdout)

    def test_formal_traditional_launcher_runs_and_aggregates_ten_seeds(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_all_per_class_fewshot.py"),
                "--datasets",
                "aid",
                "--shots",
                "1",
                "--models",
                "resnet18",
                "--dry-run",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("run_strict_baselines.py"), 10)
        for seed in range(42, 52):
            self.assertIn(f"--seed {seed} ", completed.stdout)
        self.assertEqual(completed.stdout.count("aggregate_paper_runs.py"), 1)


if __name__ == "__main__":
    unittest.main()
