#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-interactive-algorithm"

echo "Testing ToothFairy3 Interactive-Segmentation algorithm..."

# Define paths
TEST_DATA_DIR="/tmp/ToothFairy-toothfairy3/ToothFairy3"
IMAGES_DIR="$TEST_DATA_DIR/imagesTr"
CLICKS_DIR="$TEST_DATA_DIR/clicksTr"
RESULTS_DIR="/tmp/ToothFairy-toothfairy3/ToothFairy3/test-results"
ALGORITHM_OUTPUT="$RESULTS_DIR/algorithm-output"

# Create test directories
mkdir -p "$SCRIPTPATH/test/input/images/cbct"
mkdir -p "$SCRIPTPATH/test/output/images/iac-segmentation"
mkdir -p "$SCRIPTPATH/test/output/metadata"
mkdir -p "$ALGORITHM_OUTPUT"

# Copy test images and associated clicks to input directory
echo "Copying test image-click pairs from $IMAGES_DIR and $CLICKS_DIR..."

for image_file in "$IMAGES_DIR"/*.nii.gz; do
    if [ -f "$image_file" ]; then
        base_name=$(basename "$image_file" .nii.gz)
        base_name="${base_name%_0000}"

        for i in $(seq 0 5); do
            new_image_name="${base_name}_${i}.nii.gz"
            new_click_name="iac_clicks_${base_name}_${i}.json"

            cp "$image_file" "$SCRIPTPATH/test/input/images/cbct/$new_image_name"

            click_path="$CLICKS_DIR/$base_name/${i}_clicks.json"
            if [ -f "$click_path" ]; then
                cp "$click_path" "$SCRIPTPATH/test/input/$new_click_name"
                echo "  Copied: $new_image_name + $new_click_name"
            else
                echo "  WARNING: Missing click file $click_path, skipping this pair"
            fi
        done
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
