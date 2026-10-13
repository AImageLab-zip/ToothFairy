#!/usr/bin/env python3
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or set(payload) != {"report"}:
    raise ValueError("Output must be a JSON object containing only 'report'")
if not isinstance(payload["report"], str) or not payload["report"].strip():
    raise ValueError("Output report must be a non-empty string")
print(json.dumps({"verified": str(path), "characters": len(payload["report"])}))
