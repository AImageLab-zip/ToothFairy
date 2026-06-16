#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

SOURCE_DIR=${1:-"/home/llumetti/Downloads/toothfairy4-cbct---raw"}
OUTPUT_TARBALL=${2:-"$SCRIPT_DIR/evaluationgroundtruth.tar.gz"}
REFERENCE_DIR="reports_en"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Ground-truth source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

OUTPUT_PARENT=$(dirname -- "$OUTPUT_TARBALL")
mkdir -p "$OUTPUT_PARENT"

STAGING_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

echo "=+= Creating ground-truth tarball"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_TARBALL"

num_reports=0
for case_dir in "$SOURCE_DIR"/*; do
  if [ ! -d "$case_dir" ]; then
    continue
  fi

  case_id=$(basename -- "$case_dir")
  reports_dir="$case_dir/$REFERENCE_DIR"
  if [ ! -d "$reports_dir" ]; then
    continue
  fi

  first_report=$(find "$reports_dir" -maxdepth 1 -type f -name "*.txt" -printf '%f\n' | sort | head -n 1)
  if [ -z "$first_report" ]; then
    continue
  fi

  report_count=$(find "$reports_dir" -maxdepth 1 -type f -name "*.txt" | wc -l)
  if [ "$report_count" -gt 1 ]; then
    echo "Multiple English reports for $case_id; using $first_report"
  fi

  cp "$reports_dir/$first_report" "$STAGING_DIR/$case_id.txt"
  num_reports=$((num_reports + 1))
done

if [ "$num_reports" -eq 0 ]; then
  echo "No reports found at $SOURCE_DIR/*/$REFERENCE_DIR/*.txt" >&2
  exit 1
fi

tar -czvf "$OUTPUT_TARBALL" -C "$STAGING_DIR" .

echo "=+= Done: packaged $num_reports reports"
