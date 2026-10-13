#!/usr/bin/env python3

import unittest

from report_sanitizer import rejection_reason, sanitize_report


class ReportSanitizerTest(unittest.TestCase):
    def test_filters_unsupported_numbered_findings(self) -> None:
        report = (
            "The dental midlines are coincident. "
            "Restorations are present on teeth 16 and 26. "
            "Buccal gingival recession is present on tooth 41. "
            "Crossbite is present on tooth 25."
        )
        sanitized, details = sanitize_report(report, min_unsupported_sentences=1)
        self.assertEqual(
            sanitized,
            "The dental midlines are coincident. "
            "Buccal gingival recession is present on tooth 41. "
            "Crossbite is present on tooth 25.",
        )
        self.assertEqual(details["removed_sentence_count"], 1)

    def test_retains_localized_periodontal_findings(self) -> None:
        report = (
            "Buccal gingival recessions are present on teeth 15, 24, and 41. "
            "The gingivae appear inflamed."
        )
        sanitized, details = sanitize_report(report, min_unsupported_sentences=1)
        self.assertEqual(sanitized, report)
        self.assertEqual(details["removed_sentence_count"], 0)

    def test_retains_structural_and_general_hygiene_findings(self) -> None:
        report = (
            "A deep bite is present. Visible plaque is present in the posterior sectors. "
            "The gingivae appear inflamed. Mild lower crowding is present."
        )
        sanitized, details = sanitize_report(report, min_unsupported_sentences=1)
        self.assertEqual(sanitized, report)
        self.assertEqual(details["removed_sentence_count"], 0)

    def test_retains_explicit_negative_restorative_sentence(self) -> None:
        sentence = "No restorations are present."
        self.assertIsNone(rejection_reason(sentence))

    def test_filters_non_numbered_high_risk_findings(self) -> None:
        report = (
            "Two premolars are missing in the upper arch. "
            "The molar fissures are pigmented with possible presence of caries."
        )
        sanitized, details = sanitize_report(report, min_unsupported_sentences=1)
        self.assertTrue(details["fallback_used"])
        self.assertNotIn("missing", sanitized.lower())
        self.assertNotIn("caries", sanitized.lower())

    def test_removes_exact_duplicates(self) -> None:
        report = "The Curve of Spee is increased. The Curve of Spee is increased."
        sanitized, details = sanitize_report(report, min_unsupported_sentences=1)
        self.assertEqual(sanitized, "The Curve of Spee is increased.")
        self.assertEqual(details["duplicate_sentence_count"], 1)

    def test_default_threshold_targets_only_detail_heavy_reports(self) -> None:
        report = (
            "A deep bite is present. Restorations are present on teeth 16 and 26. "
            "A crown is present on tooth 36. Tooth 24 is missing. "
            "A white spot is present on tooth 11. Tooth 23 is ectopically erupting."
        )
        sanitized, details = sanitize_report(report)
        self.assertTrue(details["activated"])
        self.assertEqual(sanitized, "A deep bite is present.")

    def test_compound_structural_sentence_is_preserved(self) -> None:
        sentence = (
            "There is a Class I molar relationship, while tooth 36 is absent from the arch."
        )
        self.assertIsNone(rejection_reason(sentence))

    def test_non_dental_absence_is_not_tooth_status(self) -> None:
        self.assertIsNone(
            rejection_reason(
                "Gingival inflammation cannot be confirmed in the absence of a clinical examination."
            )
        )
        self.assertIsNone(
            rejection_reason("There is maxillary constriction in the absence of crossbite.")
        )


if __name__ == "__main__":
    unittest.main()
