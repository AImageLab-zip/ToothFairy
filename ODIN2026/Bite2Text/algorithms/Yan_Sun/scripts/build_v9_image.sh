#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=${BITE2TEXT_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}

command -v docker >/dev/null 2>&1 || {
  echo "Docker CLI is required to build the v9 image chain." >&2
  exit 1
}
docker buildx version >/dev/null 2>&1 || {
  echo "Docker Buildx is required to build the linux/amd64 image chain." >&2
  exit 1
}

"${PROJECT_ROOT}/scripts/bootstrap_upstreams.sh"
"${PROJECT_ROOT}/task2_bite2text/ptv3_submission/do_build.sh"
"${PROJECT_ROOT}/task2_bite2text/hybrid_submission_v5/do_build.sh"
"${PROJECT_ROOT}/task2_bite2text/hybrid_submission_v9_final/do_build.sh"

docker image inspect odin2026-bite2text-hybrid-photo-test-v9 >/dev/null
echo "Built odin2026-bite2text-hybrid-photo-test-v9"
