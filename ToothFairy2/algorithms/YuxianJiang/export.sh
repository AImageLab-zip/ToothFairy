#!/usr/bin/env bash

# sh ./build.sh

docker save toothfairy_algorithm | gzip -c > ToothFairy_Algorithm.tar.gz
