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
    output_instruction,
    prompt_template_sha256,
    task_text,
)


class PromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.classes = [
            "Airport", "BareLand", "BaseballField", "Beach", "Bridge",
            "Center", "Church", "Commercial", "DenseResidential", "Desert",
        ]

    def test_legacy_prompt_is_preserved(self):
        text = task_text("aid", self.classes, 3, prompt_mode="legacy")
        self.assertIn("Boundary-aware Category Rules", text)

    def test_reference_guided_prompt_prioritizes_examples(self):
        text = task_text(
            "aid", self.classes, 3, prompt_mode="reference_guided_v1"
        )
        self.assertIn("First compare the target image", text)
        self.assertIn("Category Descriptions (secondary guidance)", text)
        self.assertIn("rather than defaulting to Center", text)
        self.assertIn("Candidate Label Set: " + ", ".join(self.classes), text)

    def test_reference_only_prompt_omits_category_descriptions(self):
        text = task_text(
            "aid", self.classes, 3, prompt_mode="reference_only_v1"
        )
        self.assertIn("Compare the target image", text)
        self.assertIn("Candidate Label Set: " + ", ".join(self.classes), text)
        self.assertNotIn("Category Descriptions", text)
        self.assertNotIn("Boundary-aware Category Rules", text)
        self.assertNotIn("rather than defaulting to Center", text)

    def test_reference_fallback_v2_uses_descriptions_only_as_tie_breakers(self):
        text = task_text(
            "aid", self.classes, 3, prompt_mode="reference_fallback_v2"
        )
        self.assertIn("primary reference evidence", text)
        self.assertIn("only as tie-breakers", text)
        self.assertIn("do not let the category descriptions override it", text)
        self.assertIn("Fallback Category Descriptions", text)
        self.assertIn("visually distinctive civic", text)
        self.assertIn("worship-specific structural evidence", text)
        self.assertIn("Commercial scenes may be compact", text)
        self.assertNotIn("Category Descriptions (secondary guidance)", text)

    def test_reference_fallback_v2_uses_revised_dense_description(self):
        classes = [
            "dense_residential", "medium_residential", "sparse_residential",
            "mobile_home_park", "commercial_area", "industrial_area",
            "parking_lot", "railway_station",
        ]
        text = task_text(
            "nwpu_fg_urban", classes, 3, prompt_mode="reference_fallback_v2"
        )
        self.assertIn("Visible streets, trees, or small yards", text)
        self.assertIn("do not by themselves make the scene medium_residential", text)
        self.assertIn("Reference Demonstrations: 3", text)

    def test_reference_fallback_v3_locks_clear_example_support(self):
        text = task_text(
            "aid", self.classes, 3, prompt_mode="reference_fallback_v3"
        )
        self.assertIn("only positive classification evidence", text)
        self.assertIn("Stage A -- demonstration-only decision", text)
        self.assertIn("Freeze the best label", text)
        self.assertIn("checks have zero positive weight", text)
        self.assertIn("Never use a check to introduce a third label", text)
        self.assertIn("Pairwise Exclusion Checks", text)
        self.assertIn("Center versus Commercial", text)
        self.assertIn("Church versus Commercial", text)
        self.assertIn("Commercial versus DenseResidential", text)

    def test_reference_fallback_v3_uses_dense_exclusion_boundaries(self):
        classes = [
            "dense_residential", "medium_residential", "sparse_residential",
            "mobile_home_park", "commercial_area", "industrial_area",
            "parking_lot", "railway_station",
        ]
        text = task_text(
            "nwpu", classes, 3, prompt_mode="reference_fallback_v3"
        )
        self.assertIn("dense_residential versus medium_residential", text)
        self.assertIn("consistent across most housing blocks", text)
        self.assertIn("Visible streets, trees, pools, water", text)
        self.assertIn("dense_residential versus mobile_home_park", text)
        self.assertIn("near-uniform size and orientation", text)
        self.assertIn("Candidate Label Set: " + ", ".join(classes), text)

    def test_reference_fallback_v3_output_rejects_semantic_rationalization(self):
        text = output_instruction(
            self.classes, prompt_mode="reference_fallback_v3"
        )
        self.assertIn("brief full-scene comparison", text)
        self.assertIn("decisive visible counterevidence", text)
        self.assertIn("Do not mention inferred land-use purpose", text)

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
                prompt_mode="paper_v1",
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

    def test_unknown_prompt_mode_fails(self):
        with self.assertRaises(ValueError):
            task_text("aid", self.classes, 3, prompt_mode="unknown")


if __name__ == "__main__":
    unittest.main()
