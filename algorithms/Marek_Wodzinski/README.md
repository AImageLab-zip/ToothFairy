# ToothFairy_MW_2023
Contribution to the ToothFairy Challenge (MICCAI 2023) by Marek Wodzinski (3rd place).

Here you can see the full source code used to train / test the proposed solution.

Only the final experiment is left (the one used for the final Docker submission).

* In order to reproduce the experiment you should:
    * Download the Toothfairy dataset [Link](https://toothfairy.grand-challenge.org/dataset/)
    * Update the [hpc_paths.py](./src/paths/hpc_paths.py) and [paths.py](./src/paths/paths.py) files.
    * Run the [parse_toothfairy.py](./src/parsers/parse_toothfairy.py)
    * Run the [elastic_toothfairy.py](./src/parsers/elastic_toothfairy.py)
    * Run the training using [run_toothfairy_trainer.py](./src/runners/run_toothfairy_trainer.py)
    * And finally use the trained model for inference using [inference.py](./src/inference/inference_toothfairy.py)

The network was trained using HPC infrastructure (PLGRID). Therefore the .slurm scripts are omitted for clarity.

Please cite the ToothFairy challenge paper (TODO) if you found the source code useful.
Please find the method description: (TODO).
