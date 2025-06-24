#!/bin/bash

# Export script for ToothFairy3 Multi-Instance-Segmentation evaluation container

DOCKER_TAG="toothfairy3-multi-instance-evaluation"

docker save $DOCKER_TAG | gzip -c > ${DOCKER_TAG}.tar.gz
