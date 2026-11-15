import argparse
import json
from pathlib import Path

import httpx

from ledgerlight.examples import numeric_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in json.loads(args.file.read_text()):
        for repetition in range(3):
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    args.gateway + "/ask",
                    json={"question": item["question"]},
                    headers={"X-Forwarded-For": f"198.18.{len(rows) // 250}.{len(rows) % 250 + 1}"},
                )
            payload = response.json()
            rows.append(
                {
                    "qid": item.get("qid"),
                    "repetition": repetition + 1,
                    "http_status": response.status_code,
                    "cited": bool(payload.get("citations")),
                    "abstained": payload.get("abstained"),
                    "numeric_check": numeric_check(
                        payload.get("answer", ""), payload.get("citations", [])
                    ),
                }
            )
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
