#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm"

echo "Testing ToothFairy3 Multi-Instance-Segmentation algorithm..."

# Define paths
TEST_DATA_DIR="/c/Users/Luca/Desktop/Projects/ToothFairy/ToothFairy3/test-data/ToothFairy3"
IMAGES_DIR="$TEST_DATA_DIR/imagesTr"
RESULTS_DIR="/c/Users/Luca/Desktop/Projects/ToothFairy/ToothFairy3/test-results"
ALGORITHM_OUTPUT="$RESULTS_DIR/algorithm-output"

# Create test directories
mkdir -p "$SCRIPTPATH/test/input/images/cbct"
mkdir -p "$SCRIPTPATH/test/output/images/oral-pharyngeal-segmentation"
mkdir -p "$SCRIPTPATH/test/output/metadata"
mkdir -p "$ALGORITHM_OUTPUT"

# Copy test images to input directory
echo "Copying test images from $IMAGES_DIR..."
counter=1
for file in "$IMAGES_DIR"/*.nii.gz; do
    if [ -f "$file" ]; then
        # Rename to ensure unique first numbers for the validator
        new_name=$(printf "%03d_0000.nii.gz" $counter)
        cp "$file" "$SCRIPTPATH/test/input/images/cbct/$new_name"
        echo "  $(basename "$file") -> $new_name"
        counter=$((counter + 1))
    fi
done

# Run the Docker container
echo "Running algorithm Docker container..."
docker run --rm \
    --memory=8g \
    -v "$SCRIPTPATH/test/input":/input \
    -v "$SCRIPTPATH/test/output":/output \
    $DOCKER_TAG

# Copy results to shared location for evaluation
echo "Copying results to shared location: $ALGORITHM_OUTPUT"
cp -r "$SCRIPTPATH/test/output/"* "$ALGORITHM_OUTPUT/"

if [ $? -eq 0 ]; then
    echo "Algorithm test completed successfully!"
    echo "Results copied to: $ALGORITHM_OUTPUT"
    echo "Ready for evaluation step."
else
    echo "Error: Algorithm test failed with exit code $?"
    exit 1
fi
