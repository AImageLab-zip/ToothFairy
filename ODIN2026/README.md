# ODIN 2026 Challenges (MICCAI 2026)

ODIN 2026 is a cluster of challenges associated with the Oral and Dental Image aNalysis Workshop ([ODIN 2026](https://odin-workshops.org/2026)) at [MICCAI 2026](https://conferences.miccai.org/2026/en/), organized by the University of Modena and Reggio Emilia in collaboration with Radboud University and Karlsruhe Institute of Technology. The challenge is hosted by grand-challenge and is part of MICCAI 2026. The cluster focuses on automatic generation of structured clinical reports from routine oral and dental imaging.

Three-dimensional imaging is now routine in dentistry and maxillofacial surgery. Cone-beam computed tomography (CBCT) supports diagnosis and surgical planning by capturing internal dental and craniofacial anatomy, while intraoral scanning (IOS) provides accurate surface geometry of crowns and gingiva. Despite the increasing availability of rich 3D and complementary 2D data, clinical reporting remains largely manual, time-consuming, and subject to inter-observer variability.

ODIN 2026 addresses this gap by benchmarking systems that transform multimodal oral imaging into clinically meaningful text reports. The cluster includes two complementary tracks:

- **Task 1 - ToothFairy4:** Maxillofacial and surgical report generation from CBCT volumes.
- **Task 2 - Bite2Text:** Orthodontic report generation from intraoral scans and intraoral photographs.

The goal is to advance clinically useful multimodal 3D-to-text and 2D/3D-to-text learning, with a strong emphasis on robustness across acquisition centers, scanners, protocols, and patient populations.

## Challenge Overview

ODIN 2026 features two distinct tracks focusing on advanced dental imaging and analysis:

### Track 1: ToothFairy4 - Maxillofacial and Surgical Report Generation from CBCT
This is the fourth edition of the ToothFairy challenge, now focusing on surgical and interventional planning workflows. ToothFairy4 mirrors the clinical decision-making process for maxillofacial interventions. Participants must generate clinically actionable reports from 3D CBCT volumes of the jaws. The expected output is comprehensive reports that describe critical clinical findings including dental status, bone quality and quantity, anatomical variants, proximity to critical structures, and procedure-related risk factors. These insights support use cases such as tooth extraction, implant placement, and other maxillofacial interventions, moving beyond segmentation to clinical decision support.

### Track 2: Bite2Text - Orthodontic Report Generation from Intraoral Scans and Photographs
Bite2Text reflects routine orthodontic diagnosis and treatment planning workflows. Participants must generate orthodontic reports from multimodal data including 3D intraoral scans and 2D intraoral photographs. The task involves multimodal reasoning over dental geometry and visual appearance, requiring participants to develop solutions that integrate both 3D geometric analysis and visual assessment. Reports should describe clinically relevant orthodontic findings such as malocclusion patterns, occlusal relationships, crowding and spacing, overjet and overbite categories, molar and canine relationships, and treatment-relevant anomalies.

## Repository Structure

- `Bite2Text/`: Contains algorithm templates, evaluation scripts, and documentation for Track 2
- `ToothFairy4/`: Contains algorithm templates, evaluation scripts, and documentation for Track 1

## Getting Started

Each track contains:
- **algorithm/**: Template code for developing your submission
- **evaluation/**: Evaluation scripts and metrics used for assessment
- **README.md**: Track-specific documentation and guidelines
- **run_full_pipeline.sh**: Script to execute the complete pipeline locally

## Challenge Information

For detailed information about:
- **Challenge rules and timeline**: Visit the [Grand-Challenge website](https://odin2026.grand-challenge.org/)
- **Datasets access and documentation**: Visit the [ToothFairy4 page](https://ditto.ing.unimore.it/toothfairy4/) and the [Bite2Text page](https://ditto.ing.unimore.it/bite2text/) on Ditto website.
- **Submission guidelines**: See track-specific README files

## Docker Submission

All algorithms must be submitted as Docker containers. Each track provides:
- Dockerfile template
- Build scripts for Linux and Windows
- Test scripts for local validation
- Example algorithms

## Contact and Support

For questions and support, please visit the challenge forum on Grand-Challenge or contact the organizers through the official challenge channels.
