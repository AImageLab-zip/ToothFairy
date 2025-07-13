# ToothFairy3 Challenge (MICCAI 2025)

This is the third edition of the ToothFairy challenge, organized by the University of Modena and Reggio Emilia in collaboration with Radboud University and Karlsruhe Institute of Technology. The challenge is hosted by grand-challenge and is part of MICCAI 2025.

## Challenge Overview

ToothFairy3 introduces two distinct tracks focusing on advanced CBCT segmentation tasks:

### Track 1: Multi-Instance-Segmentation
The goal of this task is to push the development of deep learning frameworks to segment anatomical structures in CBCTs by incrementally extending the amount of publicly available 3D-annotated CBCT scans and providing additional segmentation classes for existing and specifically introduced data. New classes include pulp chamber and root canals, incisive nerves, and the lingual foramen (45 classes in total). All of these structures, missing in the previous editions of the challenge, are essential for effective orthodontic interventions. Given the high performance of ToothFairy2 submissions in most of the classes involved, this year, our goal is
twofold: introducing more challenging structures (thin and elongated) to be segmented and favoring fast and optimized algorithm proposals by introducing time as one of the main evaluation criteria and not using it only as
tie-breaker. A good tradeoff between time and accuracy is mandatory to ensure the integration of automatic solutions in daily clinical practice.

### Track 2: Interactive-Segmentation
It is an interactive segmentation task to foster innovation in the domain, inviting participants to develop and submit click-based interactive models tailored for IAC segmentation in CBCT scans. This task also provides an
opportunity for participants to leverage and build upon emerging prompt-based foundation models, further advancing the state-of-the-art in interactive medical image segmentation. By focusing on the interactive domain,
we aim to bridge the gap between automated solutions and clinical requirements, ultimately supporting more precise surgical planning and improving patient outcomes.

## Repository Structure

- `Multi-Instance-Segmentation/`: Contains algorithm templates, evaluation scripts, and documentation for Track 1
- `Interactive-Segmentation/`: Contains algorithm templates, evaluation scripts, and documentation for Track 2

## Getting Started

Each track contains:
- **algorithm/**: Template code for developing your submission
- **evaluation/**: Evaluation scripts and metrics used for assessment
- **README.md**: Track-specific documentation and guidelines

## Challenge Information

For detailed information about:
- **Challenge rules and timeline**: Visit the [Grand-Challenge website](https://toothfairy3.grand-challenge.org/)
- **Dataset access and documentation**: Visit the [ToothFairy3 page on Ditto](https://ditto.ing.unimore.it/toothfairy3/) 
- **Submission guidelines**: See track-specific README files

## Docker Submission

All algorithms must be submitted as Docker containers. Each track provides:
- Dockerfile template
- Build scripts for Linux and Windows
- Test scripts for local validation
- Example algorithms

## Contact and Support

For questions and support, please visit the challenge forum on Grand-Challenge or contact the organizers through the official challenge channels.
