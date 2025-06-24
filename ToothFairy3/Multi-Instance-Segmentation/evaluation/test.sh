#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="toothfairy3-multi-instance-evaluation"

# Define paths
TEST_DATA_DIR="/c/Users/Luca/Desktop/Projects/ToothFairy/ToothFairy3/test-data/ToothFairy3"
LABELS_DIR="$TEST_DATA_DIR/labelsTr"
RESULTS_DIR="/c/Users/Luca/Desktop/Projects/ToothFairy/ToothFairy3/test-results"
ALGORITHM_OUTPUT="$RESULTS_DIR/algorithm-output"
EVALUATION_OUTPUT="$RESULTS_DIR/evaluation-output"

echo "Running ToothFairy3 Multi-Instance-Segmentation evaluation..."

mkdir -p "$EVALUATION_OUTPUT"

# Check if algorithm output exists
if [ ! -d "$ALGORITHM_OUTPUT" ]; then
    echo "Error: Algorithm output directory not found: $ALGORITHM_OUTPUT"
    echo "Please run the algorithm test script first."
    exit 1
fi

# Extract ground truth tarball
TARBALL_PATH="$(dirname "$SCRIPT_DIR")/evaluationgroundtruth.tar.gz"
GT_EXTRACT_DIR="$EVALUATION_OUTPUT/ground_truth"

if [ ! -f "$TARBALL_PATH" ]; then
    echo "Error: Ground truth tarball not found: $TARBALL_PATH"
    exit 1
fi

echo "Extracting ground truth tarball..."
rm -rf "$GT_EXTRACT_DIR"
mkdir -p "$GT_EXTRACT_DIR"
tar -xzf "$TARBALL_PATH" -C "$GT_EXTRACT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract ground truth tarball"
    exit 1
fi

docker run --rm \
    --platform=linux/amd64 \
    --network=none \
    --cap-drop=ALL \
    --security-opt="no-new-privileges" \
    --shm-size=128m \
    --memory=4g \
    --pids-limit=256 \
    --user=1001 \
    -v "$ALGORITHM_OUTPUT":/input/:ro \
    -v "$GT_EXTRACT_DIR":/opt/ml/input/data/ground_truth/:ro \
    -v "$EVALUATION_OUTPUT":/output/ \
    $DOCKER_TAG

echo "Evaluation completed. Check results in $EVALUATION_OUTPUT/metrics.json"
