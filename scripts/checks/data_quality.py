import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = [
        ["ruff", "check", "src", "scripts", "tests"],
        ["ruff", "format", "--check", "src", "scripts", "tests"],
        ["pytest", "-q", "--basetemp=" + str(args.output / "pytest")],
        [sys.executable, "scripts/leakage_check.py", "--static"],
        [sys.executable, "-m", "ledgerlight.fixture", "ci"],
    ]
    results = []
    for index, command in enumerate(commands):
        target = args.output / f"{index}.log"
        with target.open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ
            )
        results.append({"command": command, "exit_code": result.returncode, "log": str(target)})
        (args.output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)
        if result.returncode:
            print(target.read_text()[-8000:])
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
