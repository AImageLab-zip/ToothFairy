#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
fail=0

if rg -n -i --glob '!scripts/check_release.sh' '(\/home\/[^\/[:space:]]+\/|\/Users\/[^\/[:space:]]+\/|[A-Z]:\\Users\\[^\\[:space:]]+\\)' .; then
  echo "Release check failed: user-specific filesystem path found." >&2; fail=1
fi
if rg -n -i --glob '!scripts/check_release.sh' '([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)' .; then
  echo "Release check failed: email address or credential-shaped value found." >&2; fail=1
fi
if rg -n -i --glob '!scripts/check_release.sh' '\bAAAI\b|arXiv|under review at' .; then
  echo "Release check failed: conference/preprint disclosure found." >&2; fail=1
fi
if rg --files | rg -n '(^|/)(training|evaluation|reference_evidence|experiments?|baselines?|direct_comparison)(/|$)'; then
  echo "Release check failed: research-only path found." >&2; fail=1
fi
if find . -type l -print -quit | grep -q .; then
  echo "Release check failed: symbolic links are not allowed." >&2; find . -type l -print >&2; fail=1
fi
if find . -type f \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.ckpt' -o -iname '*.safetensors' -o -iname '*.onnx' -o -iname '*.pkl' \) -print -quit | grep -q .; then
  echo "Release check failed: model or serialized artifact found." >&2; fail=1
fi
if find . -type f -size +10M -print -quit | grep -q .; then
  echo "Release check failed: file larger than 10 MiB found." >&2; find . -type f -size +10M -print >&2; fail=1
fi
python3 - <<'PY2'
import json
from pathlib import Path
for path in Path('.').rglob('*.json'):
    json.loads(path.read_text(encoding='utf-8'))
PY2
exit "$fail"
