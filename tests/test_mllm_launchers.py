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


if __name__ == "__main__":
    unittest.main()
