from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rs_visemds.category_texts import category_descriptions_sha256
from rs_visemds.prompt_builder import (
    build_local_messages_and_images,
    category_rules_sha256,
    prompt_template_sha256,
)


class PromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.classes = [
            "Airport", "BareLand", "BaseballField", "Beach", "Bridge",
            "Center", "Church", "Commercial", "DenseResidential", "Desert",
        ]

    def test_formal_prompt_hashes_are_stable_dataset_specific_sha256(self):
        aid_hashes = (
            prompt_template_sha256("aid", self.classes),
            category_descriptions_sha256("aid", self.classes),
            category_rules_sha256("aid", self.classes),
        )
        nwpu_classes = [
            "dense_residential", "medium_residential", "sparse_residential",
            "mobile_home_park", "commercial_area", "industrial_area",
            "parking_lot", "railway_station",
        ]
        nwpu_hashes = (
            prompt_template_sha256("nwpu_urban", nwpu_classes),
            category_descriptions_sha256("nwpu_urban", nwpu_classes),
            category_rules_sha256("nwpu_urban", nwpu_classes),
        )
        self.assertTrue(all(len(value) == 64 for value in (*aid_hashes, *nwpu_hashes)))
        self.assertEqual(aid_hashes[0], prompt_template_sha256("aid", self.classes))
        self.assertNotEqual(aid_hashes, nwpu_hashes)

    def test_formal_prompt_contains_only_example_labels_not_target_label(self):
        examples = [
            {"example_label": "Airport", "example_path": "a.jpg"},
            {"example_label": "Beach", "example_path": "b.jpg"},
            {"example_label": "Bridge", "example_path": "c.jpg"},
        ]
        with patch(
            "rs_visemds.prompt_builder.load_rgb_image", side_effect=lambda path: str(path)
        ):
            messages, images = build_local_messages_and_images(
                data_root=Path("data"),
                target_path="secret_target.jpg",
                dataset="aid",
                class_order=self.classes,
                examples=examples,
            )
        text = "\n".join(
            part.get("text", "")
            for part in messages[1]["content"]
            if part.get("type") == "text"
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Candidate Label Set: " + ", ".join(self.classes), text)
        self.assertIn("Boundary-aware Category Rules", text)
        self.assertIn("Visual-Semantic Demonstrations: 3", text)
        self.assertIn("Ground-truth label: Airport", text)
        self.assertIn("Target Input: the next image is the unlabeled target image", text)
        self.assertNotIn("secret_target.jpg", text)
        self.assertEqual(len(images), 4)


if __name__ == "__main__":
    unittest.main()
