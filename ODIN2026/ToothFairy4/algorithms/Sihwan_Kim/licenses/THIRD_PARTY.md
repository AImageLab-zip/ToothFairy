# Third-party attribution

The repository's code licence covers original REPGEN contributions. It does
not replace any licence applying to an upstream implementation or dependency.

| Project | Source | Licence |
| --- | --- | --- |
| U-Mamba2 backbone implementation | https://github.com/zhiqin1998/U-Mamba2, revision `2046d29785087b656ca69fa02dd40e43e69cfb42` | CC BY-NC 4.0 |
| nnU-Net | https://github.com/MIC-DKFZ/nnUNet | Apache-2.0 |
| Mamba / mamba-ssm | https://github.com/state-spaces/mamba | Apache-2.0 |
| causal-conv1d | https://github.com/Dao-AILab/causal-conv1d | BSD-3-Clause |

The reference Mamba2 execution helper adapts operations from the Mamba
implementation. Copyright and licence notices from upstream packages remain
applicable. The Docker build obtains the pinned backbone and installs its
dependencies with their package licence files. The inference compatibility
class preserves the submitted loader name and uses nnU-Net's network builder.

Training data have separate terms: ToothFairy3 is CC BY-NC-SA 4.0 and DOLCHID
version 1 is CC BY 4.0. Neither dataset is redistributed. The licence selected
for our trained weights is declared in `licenses/WEIGHTS.txt`; it is not a claim
that dataset licence terms automatically determine the legal status of weights.
