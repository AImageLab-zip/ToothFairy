#!/bin/bash
# ToothFairy3 Multi-Instance-Segmentation Complete Pipeline
# Runs algorithm on test data, then evaluates against ground truth

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "=== ToothFairy3 Multi-Instance-Segmentation Pipeline ==="
echo ""

# Step 1: Run Algorithm
echo "Step 1: Running Algorithm..."
cd "$SCRIPT_DIR/algorithm"
./build.sh
./test.sh

if [ $? -ne 0 ]; then
    echo "Error: Algorithm step failed. Pipeline aborted."
    exit 1
fi

echo ""
echo "Step 2: Running Evaluation..."
cd "$SCRIPT_DIR/evaluation"
./build.sh
./test.sh

if [ $? -ne 0 ]; then
    echo "Error: Evaluation step failed."
    exit 1
fi

echo ""
echo "=== Pipeline Completed Successfully ==="
echo "Results available in:"
echo "  - Algorithm Output: /c/Users/Luca/Desktop/Projects/ToothFairy/ToothFairy3/test-results/algorithm-output"
echo "  - Evaluation Output: /c/Users/Luca/Desktop/Projects/ToothFairy/ToothFairy3/test-results/evaluation-output"
