# ToothFairy3 Interactive-Segmentation Algorithm Template

This is a template that you can use to develop and test your algorithm.

To run it, you'll need to install [docker](https://docs.docker.com/engine/install/) and [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

First of all, you have to clone this repository and `cd` in the algorithm directory:
```
git clone https://github.com/AImageLab-zip/ToothFairy.git
cd ToothFairy/ToothFairy3/Interactive-Segmentation/algorithm
```

All the code has been tested on both Linux (Ubuntu 18.04.6) and Windows 11

## Download Debug Images
To debug your algorithm and test it locally with demo images from the training set, download the `test` folder from this [link](https://drive.google.com/drive/folders/1Dd8B_p-hAE2xhMRwafmkg2BaDAWDW0xH?usp=sharing) and put it here.

## Testing Your Algorithm
Inside the `process.py` file you have to
add all the steps required by your algorithm. A simple example is already
provided.

When you are ready, check that everything works properly by running `./test.sh`. **IMPORTANT**: Update the local paths to `imagesTr` and `clicksTr` in the `./test.sh` script first!


## Submit Your Algorithm
Once you have checked that everything works properly using `test.sh`, you are ready to export your algorithm into a Docker container using `./export.sh` and ship it to Grand-Challenge from the [submission page](https://toothfairy3.grand-challenge.org/evaluation/debugging-phase/submissions/create/) of the challenge. Be carefull because you have a limited amount of submissions, evaluate you algorithm locally before submitting!

## Simulating Clicks
Use the `simulate_clicks.py` to simulate clicks for your own algorithms. This script is also the script used to simulate all clicks for our training and test cases.

This script accepts the following arguments:

| Argument | Required | Description |
|----------|----------|-------------|
| `-i`, `--input_label` | ✅ Yes | **Path to the .mha or nii.gz label file**. This is the input label file that the script will process. . |
| `--debug_output` | ❌ No | **Output path for click visualization**. If provided, the script will save a visual representation of the clicks as Gaussian heatmaps in a `.nii.gz` file. |
| `--gc_archive_output` | ❌ No | **Output path for Grand Challenge archives**. Optional path where the script will store archives of 0–5 clicks formatted for upload to the Grand Challenge platform. |
| `--json_output` | ✅ Yes | **Output path for JSON**. Required path where the script will write all click data in a JSON format containing all IAC clicks. |

### Example Usage

```bash
python your_script.py \
  -i path/to/input_label.mha \
  --json_output path/to/output.json \
  --debug_output path/to/debug.png \
  --gc_archive_output path/to/archive.zip
```
