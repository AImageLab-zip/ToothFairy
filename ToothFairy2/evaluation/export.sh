#!/usr/bin/env bash

./build.sh

docker save toothfairy_evaluation | gzip -c > ToothFairy_Evaluation.tar.gz
