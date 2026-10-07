#!/usr/bin/env python3
"""
scripts/pipe_to_backend.py
==========================
Reads JSONL flow records from standard input (emitted by Rust network-analyzer)
and posts them in real time to the FastAPI /api/classify endpoint.

Usage:
  LD_LIBRARY_PATH=./lib/usr/lib/x86_64-linux-gnu ./target/release/network-analyzer --interface lo | python3 scripts/pipe_to_backend.py
  cat live_flows.jsonl | python3 scripts/pipe_to_backend.py
"""

import sys
import json
import urllib.request
import urllib.error

BACKEND_URL = "http://localhost:8000/api/classify"

def main():
    count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        payload = json.dumps({"flow": record}).encode("utf-8")
        req = urllib.request.Request(
            BACKEND_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                count += 1
                fid = record.get("flow_id", f"flow-{count}")
                print(f"[LIVE BRIDGE] Ingested flow #{count} ({fid}) into NIDS backend", file=sys.stderr)
        except Exception as e:
            print(f"[LIVE BRIDGE] Error forwarding flow: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
