# ToothFairy3 - Multi-Instance-Segmentation Algorithm Template

This is a template that you can use to develop and test your algorithm.

To run it, you'll need to install [docker](https://docs.docker.com/engine/install/) and [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

First of all, you have to clone this repository and `cd` in the algorithm directory:
```
git clone https://github.com/AImageLab-zip/ToothFairy.git
cd ToothFairy/ToothFairy3/Multi-Instance-Segmentation/algorithm
```

All the code has been tested on both Linux (6.2.8-arch1-1) and Windows 11

## Testing Your Algorithm
Inside the `process.py` file you have to
add all the steps required by your algorithm. A simple example is already
provided.

When you are ready, check that everything works properly by running `./test.sh`.


## Submit Your Algorithm
Once you have checked that everything works properly using `test.sh`, you are ready to export your algorithm into a Docker container using `./export.sh` and ship it to Grand-Challenge from the [submission page](https://toothfairy3.grand-challenge.org/evaluation/debugging-phase/submissions/create/) of the challenge. Be carefull because you have a limited amount of submissions, evaluate you algorithm locally before submitting!


