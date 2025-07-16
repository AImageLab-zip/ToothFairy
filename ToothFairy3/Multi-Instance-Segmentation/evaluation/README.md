# ToothFairy3 - Multi-Instance-Segmentation Evaluation

This is the source code for the evaluation of the submitted algorithms on the Grand-Challenge platform. The exact code for computing the Dice coefficient and the Hausdorff distance 95% can be found inside the evaluation.py file, under the `compute_dice()` and `compute_hd95()` functions.

If you found any problem related to such algorithms, please report it to us by opening an issue in this repository.

You can evaluate your algorithm locally by running `./test` or the `run-pipeline.sh` in the parent directory.
Ensure to modify some hardcoded path with the correct ones before running the scripts
