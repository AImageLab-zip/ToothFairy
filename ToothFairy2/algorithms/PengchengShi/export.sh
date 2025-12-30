#!/usr/bin/env bash

./build.sh

docker save toothfairy_algorithm | gzip -c > ToothFairy_Algorithm.tar.gz
