# Define the Docker tag and construct the output file name
$DOCKER_TAG = "example-evaluation-test-phase"
$OUTPUT_FILE = "$DOCKER_TAG.tar.gz"

# Save the Docker image to a tar file
$TEMP_TAR = "$DOCKER_TAG.tar"
docker save $DOCKER_TAG -o $TEMP_TAR

# Path to the 7-Zip executable
$SevenZipPath = "C:\Program Files\7-Zip\7z.exe"

# Check if the 7-Zip executable exists
if (-Not (Test-Path $SevenZipPath)) {
    Write-Error "7z.exe not found at path $SevenZipPath. Please check the installation."
    exit 1
}

# Compress the tar file to a .gz file using 7-Zip
& $SevenZipPath a -tgzip $OUTPUT_FILE $TEMP_TAR

# Remove the temporary tar file
Remove-Item $TEMP_TAR

Write-Host "Docker image saved and compressed to $OUTPUT_FILE"
