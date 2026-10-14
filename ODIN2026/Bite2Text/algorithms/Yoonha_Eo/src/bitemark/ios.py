"""Public IOS feature interpretation with randomized placeholder thresholds."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence

from .schema import Finding


def ordered_label(value: float, cuts: Sequence[float], labels: Sequence[str]) -> str:
    if len(labels) != len(cuts) + 1:
        raise ValueError("Ordered labels must contain one more item than cuts")
    if list(cuts) != sorted(cuts):
        raise ValueError("Cuts must be increasing")
    return str(labels[bisect_right([float(cut) for cut in cuts], float(value))])


def missing_teeth(present: Sequence[int], expected: Sequence[int]) -> list[int]:
    present_set = {int(value) for value in present}
    return [int(value) for value in expected if int(value) not in present_set]


def findings_from_synthetic_ios(
    features: Mapping[str, object], ios_config: Mapping[str, object], labels: Mapping[str, object]
) -> list[Finding]:
    molar_cuts = ios_config["molar_cuts"]
    canine_cuts = ios_config["canine_cuts"]
    assessability = float(ios_config["canine_assessability_score"])
    findings: list[Finding] = []

    for side in ("right", "left"):
        findings.append(
            Finding(
                "molar",
                ordered_label(float(features[f"{side}_molar_score"]), molar_cuts, labels["molar"]),
                "IOS geometry",
                side=side,
            )
        )
        if float(features[f"{side}_canine_assessability"]) < assessability:
            canine_value = "Not assessable"
        else:
            canine_value = ordered_label(
                float(features[f"{side}_canine_score"]), canine_cuts, labels["canine"]
            )
        findings.append(Finding("canine", canine_value, "IOS geometry", side=side))

    findings.extend(
        [
            Finding(
                "anterior_bite",
                ordered_label(
                    float(features["anterior_bite_score"]),
                    ios_config["anterior_bite_cuts"],
                    labels["anterior_bite"],
                ),
                "IOS geometry",
            ),
            Finding(
                "transverse_bite",
                ordered_label(
                    float(features["transverse_bite_score"]),
                    ios_config["transverse_bite_cuts"],
                    labels["transverse_bite"],
                ),
                "IOS geometry",
            ),
            Finding("overjet", str(features["overjet_label"]), "IOS point-cloud head"),
            Finding("curve_of_spee", str(features["spee_label"]), "IOS point-cloud head"),
            Finding("curve_of_wilson", str(features["wilson_label"]), "IOS point-cloud head"),
        ]
    )

    absent = missing_teeth(
        features["present_fdis"], ios_config["expected_fdi_without_third_molars"]
    )
    findings.append(
        Finding("missing_teeth", ", ".join(map(str, absent)) if absent else "None", "IOS FDI set")
    )
    return findings
