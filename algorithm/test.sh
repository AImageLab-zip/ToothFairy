#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="4g"

docker volume create toothfairy_algorithm-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v toothfairy_algorithm-output-$VOLUME_SUFFIX:/output/ \
        toothfairy_algorithm



docker run --rm \
        -v toothfairy_algorithm-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/results.json | python -m json.tool

docker run --rm \
        -v toothfairy_algorithm-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim ls -lah /output/images/inferior-alveolar-canal/

cp -r /var/lib/docker/volumes/toothfairy_algorithm-output-$VOLUME_SUFFIX/ output
chown llumetti:llumetti -R output

docker volume rm toothfairy_algorithm-output-$VOLUME_SUFFIX
