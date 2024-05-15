#!/usr/bin/env bash

./build.sh

docker save toothfairy_algorithm | pigz -c > ToothFairy_Algorithm.tar.gz