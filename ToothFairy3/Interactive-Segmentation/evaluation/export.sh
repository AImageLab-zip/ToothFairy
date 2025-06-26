#!/bin/bash

# Export script for ToothFairy3 Interactive-Segmentation evaluation container

DOCKER_TAG="toothfairy3-interactive-evaluation"
OUTPUT_FILE="$DOCKER_TAG.tar.gz"

echo "Exporting Docker image: $DOCKER_TAG"

# Save the Docker image to a tar file and compress it
docker save $DOCKER_TAG | gzip -c > $OUTPUT_FILE

if [ $? -eq 0 ]; then
    echo "Docker image saved and compressed to $OUTPUT_FILE"
    echo "Export completed successfully: $OUTPUT_FILE"
else
    echo "Error: Failed to save Docker image"
    exit 1
fi
