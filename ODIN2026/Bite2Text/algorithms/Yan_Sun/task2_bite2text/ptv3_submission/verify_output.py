#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if list(payload) != ["report"]:
    raise SystemExit(f"Unexpected keys: {list(payload)}")
if not isinstance(payload["report"], str) or not payload["report"].strip():
    raise SystemExit("report must be a non-empty string")
print(json.dumps({"schema_ok": True, "report_chars": len(payload["report"])}, sort_keys=True))
