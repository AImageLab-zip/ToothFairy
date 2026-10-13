from __future__ import annotations

import unittest

from task2_bite2text.ptv3_finetune.report_renderer import render_report


class ReportRendererTest(unittest.TestCase):
    def test_f5535_oracle_structure(self) -> None:
        report = render_report(
            {
                "right_molar_relation": "class_i",
                "right_canine_relation": "not_assessable",
                "left_molar_relation": "class_iii",
                "left_canine_relation": "not_assessable",
                "overjet": "normal",
                "vertical_relation": "normal",
                "midline_relation": "deviated",
                "crossbite": "none",
                "upper_crowding": "none",
                "lower_crowding": "mild",
                "curve_spee": "normal",
                "curve_wilson": "increased",
            }
        )
        self.assertIn("Class I molar", report)
        self.assertIn("Class III molar", report)
        self.assertIn("not assessable", report)
        self.assertIn("no crossbite", report)
        self.assertIn("curve of Spee is within normal limits", report)
        self.assertIn("mild crowding in the lower arch", report)

    def test_bilateral_compaction(self) -> None:
        report = render_report(
            {
                "right_molar_relation": "class_ii_full",
                "right_canine_relation": "class_ii_full",
                "left_molar_relation": "class_ii_full",
                "left_canine_relation": "class_ii_full",
                "overjet": "increased",
            }
        )
        self.assertEqual(
            report,
            "Sagittally, there is a bilateral full Class II molar and canine relationship, with overjet increased.",
        )


if __name__ == "__main__":
    unittest.main()
