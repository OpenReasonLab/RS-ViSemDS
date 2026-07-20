from __future__ import annotations

import unittest

from run_knn_totalshot_mllm import build_task_prompt as build_knn_prompt
from run_random_fewshot_mllm import build_task_prompt as build_random_prompt
from run_zero_shot_mllm import SYSTEM_PROMPT, build_prompt as build_zero_prompt


CLASSES = ["Airport", "BareLand", "Beach"]


class ManuscriptPromptTests(unittest.TestCase):
    def test_shared_system_prompt_blocks_filename_and_metadata(self) -> None:
        self.assertIn("Do not use filenames, metadata", SYSTEM_PROMPT)
        self.assertIn("Select exactly one label", SYSTEM_PROMPT)

    def test_zero_shot_minimal_has_no_category_guidance(self) -> None:
        prompt = build_zero_prompt("aid", CLASSES, "minimal")
        self.assertIn("Task Instruction: Classify the target", prompt)
        self.assertIn("Candidate Label Set:", prompt)
        self.assertNotIn("Dataset-specific considerations", prompt)
        self.assertNotIn("Demonstrations", prompt)

    def test_random_prompt_matches_manuscript_role(self) -> None:
        prompt = build_random_prompt("aid", CLASSES, 10, "minimal")
        self.assertIn("randomly sampled labeled images", prompt)
        self.assertIn("Randomly Sampled Demonstrations: 10", prompt)
        self.assertNotIn("Dataset-specific considerations", prompt)

    def test_knn_prompt_matches_manuscript_role(self) -> None:
        prompt = build_knn_prompt("aid", CLASSES, 3, "minimal")
        self.assertIn("visually retrieved labeled images", prompt)
        self.assertIn("Visually Retrieved Demonstrations: 3", prompt)
        self.assertNotIn("RemoteCLIP", prompt)
        self.assertNotIn("Dataset-specific considerations", prompt)


if __name__ == "__main__":
    unittest.main()
