# ToothFairy Algorithm
This is a template that you can use to develop and test your algorithm.

To run it, you'll need to install [docker](https://docs.docker.com/engine/install/) and [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

First of all, you have to clone this repository and `cd` in the algorithm directory:
```
git clone https://github.com/AImageLab-zip/ToothFairy.git
cd ToothFairy/algorithm
```

All the code has been tested on Linux (6.2.8-arch1-1)

## Testing Your Algorithm
To test your algorithm, you can use the samples provided in the `test` folder,
which are already converted to the `.mha` format that grand-challenge use
behind the scenes. If you wish to load more test samples, you will have to
convert all the `data.npy` to `<patient-name>.mha`. This conversion can be made
using SimpleITK library for python. 

Inside the `process.py` file you have to
add all the steps required by your algorithm. A simple example is already
provided: A `SimpleNet` is declared (a `torch.nn.Module`) and inside the
`predict()` function I've already took care of converting the `SimpleITK.Image`
input to a `torch.tensor`. and the output from a `torch.tensor` back to a
`SimpleITK.Image`. Feel free to modify this script but keep in mind that
GrandChallenge will give you *a single image* as input and wants *a single
image* as output, both as a `SimpleITK.Image`.

When you are ready, check that everything works properly by running `./test.sh`.


## Submit Your Algorithm
Once you have checked that everything works properly using `test.sh`, you are ready to export your algorithm into a docker container using `./export.sh` and ship it to Grand-Challenge from the [submission page](https://toothfairy.grand-challenge.org/evaluation/challenge/submissions/create/) of the challenge. Be carefull because you have a limited amount of submissions: 15 for the *Prelimaniry Test Phase*, 2 for the *Final Test Phase*. 


