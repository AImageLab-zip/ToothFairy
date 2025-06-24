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

# Cleanup function to fix permissions after container run
cleanup_permissions() {
    echo "Cleaning up permissions..."
    # Ensure permissions are set correctly on the output
    # This allows the host user to access and handle these files
    docker run --rm \
        --platform=linux/amd64 \
        --quiet \
        --volume "$EVALUATION_OUTPUT":/output \
        --entrypoint /bin/sh \
        $DOCKER_TAG \
        -c "chmod -R -f o+rwX /output/* || true"
}

# Create evaluation output directory
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

if [ $? -eq 0 ]; then
    cleanup_permissions
    echo "Evaluation completed successfully!"
    METRICS_FILE="$EVALUATION_OUTPUT/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "Results saved to: $METRICS_FILE"
        echo ""
        echo "=== Evaluation Results Preview ==="
        # Try to extract some basic metrics if jq is available
        if command -v jq >/dev/null 2>&1; then
            if jq -e '.aggregates.DiceCoefficient_Average_All.mean' "$METRICS_FILE" >/dev/null 2>&1; then
                echo "Average Dice Coefficient: $(jq -r '.aggregates.DiceCoefficient_Average_All.mean' "$METRICS_FILE")"
            fi
            if jq -e '.aggregates.HausdorffDistance95_Average_All.mean' "$METRICS_FILE" >/dev/null 2>&1; then
                echo "Average HD95: $(jq -r '.aggregates.HausdorffDistance95_Average_All.mean' "$METRICS_FILE")"
            fi
        else
            echo "Install 'jq' for metrics preview"
        fi
        echo "==================================="
    else
        echo "Warning: Metrics file not found at expected location: $METRICS_FILE"
    fi
    echo "Results available in: $EVALUATION_OUTPUT"
else
    echo "Error: Evaluation failed with exit code $?"
    exit 1
fi
