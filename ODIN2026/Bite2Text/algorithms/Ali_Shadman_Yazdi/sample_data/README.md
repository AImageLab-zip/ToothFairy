# Sample case: F5535

> **The files listed below are not yet in the repository.** They will be added
> once redistribution of this case is confirmed.

One complete case from the [Bites2Text](https://ditto.ing.unimore.it/bite2text/)
dataset, for running and testing the pipeline end to end.

| file | what |
| --- | --- |
| `ios_upper.stl`, `ios_lower.stl` | upper and lower intraoral scans (binary STL, mm, shared occlusion frame) |
| `intraoral_1.jpg` … `intraoral_5.jpg` | the five-view photo series: front, right buccal, left buccal, lower occlusal, upper occlusal |
| `reference_report_photo_en.txt` | clinician report written from the photographs (the family the challenge scores against) |
| `reference_report_ios_en.txt` | clinician report written from the scans |
| `expected_output.json` | this repository's output on this case (`python predict.py ... --seed 0`) |

**F5535 is held out.** It belongs to the frozen 198-subject test split and was
never used for training, validation, template mining or any threshold choice.
None of the 3D networks was trained on Bites2Text at all. Its reference reports
are therefore a fair check, not a memorised answer.

The photographs have had all EXIF/XMP metadata removed losslessly
(`jpegtran -copy none`); pixel data is unchanged.

## License and citation

This folder is redistributed from the Bites2Text dataset under
**[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**:
non-commercial use only, attribution required, derivatives under the same
license. Data release is covered by local approvals from the University of
Ferrara (No. 262/2025/Oss/UniFe). If you use it, cite:

> L. Lumetti, F. Rizzo, F. Cremonini, E. Candeloro, L. Lombardo, C. Grana, F. Bolelli.
> *Do Multimodal LLMs Understand Intraoral Dental Data? Dataset, Platform, and Baselines.*
> European Conference on Computer Vision (ECCV), 2026.

The full dataset (≈1000 patients) is available after registration at
<https://ditto.ing.unimore.it/bite2text/>.
