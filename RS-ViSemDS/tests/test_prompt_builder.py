from __future__ import annotations

import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rs_visemds.prompt_builder import output_instruction, task_text


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

    def test_unknown_prompt_mode_fails(self):
        with self.assertRaises(ValueError):
            task_text("aid", self.classes, 3, prompt_mode="unknown")


if __name__ == "__main__":
    unittest.main()
