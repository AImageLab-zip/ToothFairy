from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("parse_bite2text_reports.py")
SPEC = importlib.util.spec_from_file_location("bite2text_parser", MODULE_PATH)
assert SPEC and SPEC.loader
PARSER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PARSER
SPEC.loader.exec_module(PARSER)


class SagittalParserV3Test(unittest.TestCase):
    def labels(self, text: str) -> dict[str, str | None]:
        return PARSER.parse_report(text)["labels"]

    def assert_no_relation_conflict(self, parsed: dict) -> None:
        self.assertFalse(
            any(
                warning.startswith("conflict_")
                and ("molar_relation" in warning or "canine_relation" in warning)
                for warning in parsed["warnings"]
            ),
            parsed["warnings"],
        )

    def test_f5535_target_first_and_not_assessable(self) -> None:
        parsed = PARSER.parse_report(
            "Sagittally, there is a right molar Class I, canine not assessable, "
            "and a left molar Class III, canine not assessable, with a slightly reduced overjet."
        )
        labels = parsed["labels"]
        self.assertEqual(labels["right_molar_relation"], "class_i")
        self.assertEqual(labels["left_molar_relation"], "class_iii")
        self.assertEqual(labels["right_canine_relation"], "not_assessable")
        self.assertEqual(labels["left_canine_relation"], "not_assessable")
        self.assert_no_relation_conflict(parsed)

    def test_compact_right_left_target_first(self) -> None:
        labels = self.labels(
            "Sagittally, there is a right molar and canine Class I and a left molar "
            "and canine Class II edge-to-edge, with overjet within normal limits."
        )
        self.assertEqual(labels["right_molar_relation"], "class_i")
        self.assertEqual(labels["right_canine_relation"], "class_i")
        self.assertEqual(labels["left_molar_relation"], "class_ii_edge_to_edge")
        self.assertEqual(labels["left_canine_relation"], "class_ii_edge_to_edge")

    def test_bilateral_target_first(self) -> None:
        labels = self.labels(
            "From a sagittal standpoint, there is a bilateral molar and canine Class I."
        )
        for side in ("right", "left"):
            self.assertEqual(labels[f"{side}_molar_relation"], "class_i")
            self.assertEqual(labels[f"{side}_canine_relation"], "class_i")

    def test_right_and_left_class_first(self) -> None:
        labels = self.labels(
            "The patient presents a right and left Class I molar and canine relationship."
        )
        for side in ("right", "left"):
            self.assertEqual(labels[f"{side}_molar_relation"], "class_i")
            self.assertEqual(labels[f"{side}_canine_relation"], "class_i")

    def test_class_first_split_targets(self) -> None:
        labels = self.labels(
            "There is a Class I molar and Class III canine relationship on the right, "
            "whereas on the left there is a Class III molar and Class I canine relationship."
        )
        self.assertEqual(labels["right_molar_relation"], "class_i")
        self.assertEqual(labels["right_canine_relation"], "class_iii")
        self.assertEqual(labels["left_molar_relation"], "class_iii")
        self.assertEqual(labels["left_canine_relation"], "class_i")

    def test_unqualified_relation_is_bilateral(self) -> None:
        labels = self.labels(
            "Sagittally, there is a Class I molar relationship and a bilateral canine "
            "end-to-end Class II, with increased overjet."
        )
        self.assertEqual(labels["right_molar_relation"], "class_i")
        self.assertEqual(labels["left_molar_relation"], "class_i")
        self.assertEqual(labels["right_canine_relation"], "class_ii_edge_to_edge")
        self.assertEqual(labels["left_canine_relation"], "class_ii_edge_to_edge")

    def test_bilateral_not_assessable(self) -> None:
        parsed = PARSER.parse_report(
            "Sagittally, the molar class cannot be assessed on either the right or the left "
            "due to missing teeth, whereas there is a bilateral full Class II canine relationship."
        )
        labels = parsed["labels"]
        self.assertEqual(labels["right_molar_relation"], "not_assessable")
        self.assertEqual(labels["left_molar_relation"], "not_assessable")
        self.assertEqual(labels["right_canine_relation"], "class_ii_full")
        self.assertEqual(labels["left_canine_relation"], "class_ii_full")
        self.assert_no_relation_conflict(parsed)

    def test_bilateral_target_scope_does_not_leak(self) -> None:
        parsed = PARSER.parse_report(
            "Sagittally, there is a weak bilateral Class I molar relationship and a bilateral "
            "edge-to-edge Class II canine relationship with increased overjet."
        )
        labels = parsed["labels"]
        self.assertEqual(labels["right_molar_relation"], "class_i")
        self.assertEqual(labels["left_molar_relation"], "class_i")
        self.assertEqual(labels["right_canine_relation"], "class_ii_edge_to_edge")
        self.assertEqual(labels["left_canine_relation"], "class_ii_edge_to_edge")
        self.assert_no_relation_conflict(parsed)

    def test_curve_predicates_are_kept_separate(self) -> None:
        labels = self.labels(
            "The curve of Spee is within normal limits and the curve of Wilson is increased."
        )
        self.assertEqual(labels["curve_spee"], "normal")
        self.assertEqual(labels["curve_wilson"], "increased")

    def test_shared_curve_predicate(self) -> None:
        labels = self.labels("The curves of Spee and Wilson are accentuated.")
        self.assertEqual(labels["curve_spee"], "increased")
        self.assertEqual(labels["curve_wilson"], "increased")

    def test_slash_crowding_severity(self) -> None:
        labels = self.labels(
            "There is mild/moderate crowding in the upper arch and moderate/severe crowding in the lower arch."
        )
        self.assertEqual(labels["upper_crowding"], "mild-to-moderate")
        self.assertEqual(labels["lower_crowding"], "moderate-to-severe")


if __name__ == "__main__":
    unittest.main()
