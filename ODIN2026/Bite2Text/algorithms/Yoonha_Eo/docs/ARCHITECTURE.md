# Public architecture map

> Historical synthetic-skeleton documentation, retained for compatibility. For
> the complete public final2 implementation and weights, see the root README.
> References below to private components describe the old release, not final2.

This document describes interfaces and information flow only. It omits private weights, validation-selected parameters, dataset locations, and corpus-optimized report wording.

| Finding | Primary input | Representation | Public decision interface |
|---|---|---|---|
| IOP view | all photographs | image-level features | globally constrained five-view assignment |
| Upper/lower crowding | assigned occlusal photographs | arch image | private classifier output |
| Dental midline | assigned frontal photograph | FDI incisor masks | normalized mask-center displacement |
| Missing teeth | upper/lower IOS | detected FDI set | absent expected identifiers |
| Molar relationship | posterior IOS teeth | paired surface statistic | ordered placeholder thresholds |
| Canine relationship | canine masks and lower landmarks | tip/landmark statistic | assessability gate and ordered thresholds |
| Anterior bite | anterior IOS teeth | paired surface statistic | ordered placeholder thresholds |
| Transverse bite | posterior IOS teeth | paired transverse statistic | ordered placeholder thresholds |
| Overjet | both IOS arches | canonicalized dual-arch point cloud | private point-cloud classifier |
| Curve of Spee | IOS dentition | shared FDI tooth cloud | private point-cloud classifier |
| Curve of Wilson | IOS dentition | shared FDI tooth cloud | private point-cloud classifier |

## IOP branch

The private view model produces one probability vector per photograph. The public router solves a maximum-score one-to-one assignment so exactly one left, right, upper, frontal, and lower view is retained. Crowding heads consume the assigned upper and lower occlusal views. The frontal image is segmented into FDI-labeled incisor masks, from which a normalized inter-arch midline displacement can be measured.

## IOS branch

A private 3D backend produces FDI tooth instances and dental landmarks once per case. Item-specific modules then reuse this common anatomy. Local inter-arch relationships are represented by surface-derived scalar features. Global arch-shape findings are delegated to separately trained point-cloud heads. Missing teeth are derived from FDI presence.

## Report generation

Each module returns a structured finding. The report generator maps these findings to a fixed vocabulary and fixed order. It performs no language-model inference and does not expose the private corpus-optimized realization rules.

## Public-release limitation

Every numeric threshold in the public configuration is random. The configuration is deliberately rejected for ordinary execution and is usable only for the synthetic example with an explicit override.
