#!/usr/bin/env python3
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if set(payload) != {"report"} or not isinstance(payload["report"], str) or not payload["report"].strip():
    raise SystemExit(f"Invalid Bite2Text output: {payload!r}")
print(json.dumps({"valid": True, "characters": len(payload["report"])}, sort_keys=True))

