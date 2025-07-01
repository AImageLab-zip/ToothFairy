# Export script for ToothFairy3 Multi-Instance-Segmentation evaluation container

$ErrorActionPreference = "Stop"
$DOCKER_TAG = "toothfairy3-multi-instance-evaluation"
$OUTPUT_FILE = "$DOCKER_TAG.tar.gz"

# Find Docker executable
$DOCKER_EXE = "docker"
if (-Not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $DOCKER_PATH = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    if (Test-Path $DOCKER_PATH) {
        $DOCKER_EXE = $DOCKER_PATH
        Write-Host "Using Docker at: $DOCKER_PATH"
    } else {
        Write-Error "Docker not found. Please ensure Docker Desktop is installed and running."
        exit 1
    }
}

Write-Host "Exporting Docker image: $DOCKER_TAG"

# Save the Docker image to a tar file
$TEMP_TAR = "$DOCKER_TAG.tar"
& $DOCKER_EXE save $DOCKER_TAG -o $TEMP_TAR

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to save Docker image"
    exit 1
}

# Path to the 7-Zip executable
$SevenZipPath = "C:\Program Files\7-Zip\7z.exe"

# Check if the 7-Zip executable exists
if (-Not (Test-Path $SevenZipPath)) {
    Write-Warning "7z.exe not found at path $SevenZipPath. Trying to use built-in PowerShell compression..."
    
    # Fallback to PowerShell compression (less efficient but available)
    try {
        # Read the tar file and compress it
        $tarBytes = [System.IO.File]::ReadAllBytes($TEMP_TAR)
        $gzipStream = [System.IO.File]::Create($OUTPUT_FILE)
        $gzipCompressor = New-Object System.IO.Compression.GZipStream($gzipStream, [System.IO.Compression.CompressionMode]::Compress)
        $gzipCompressor.Write($tarBytes, 0, $tarBytes.Length)
        $gzipCompressor.Close()
        $gzipStream.Close()
        
        Write-Host "Docker image saved and compressed to $OUTPUT_FILE using PowerShell compression"
    }
    catch {
        Write-Error "Failed to compress using PowerShell: $_"
        Remove-Item $TEMP_TAR -ErrorAction SilentlyContinue
        exit 1
    }
} else {
    # Use 7-Zip for better compression
    & $SevenZipPath a -tgzip $OUTPUT_FILE $TEMP_TAR
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker image saved and compressed to $OUTPUT_FILE using 7-Zip"
    } else {
        Write-Error "Failed to compress with 7-Zip"
        Remove-Item $TEMP_TAR -ErrorAction SilentlyContinue
        exit 1
    }
}

# Remove the temporary tar file
Remove-Item $TEMP_TAR

Write-Host "Export completed successfully: $OUTPUT_FILE"
