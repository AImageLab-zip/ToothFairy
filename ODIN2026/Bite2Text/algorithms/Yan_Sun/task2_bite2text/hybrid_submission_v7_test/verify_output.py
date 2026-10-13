#!/usr/bin/env python3
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
assert set(payload) == {"report"}
assert isinstance(payload["report"], str) and payload["report"].strip()
print(json.dumps({"verified": str(path), "characters": len(payload["report"])}))
