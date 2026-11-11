import json
import subprocess
import sys
import tarfile

from ledgerlight.common import digest, path, read_json, write_json


def fetch(key):
    item = read_json("services.lock.json")[key]
    if key == "qwen":
        destination = path("models/qwen")
    else:
        destination = path(".tools/downloads")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / item["name"]
    expected = item["digest"].split(":")[-1]
    if not archive.exists() or digest(archive) != expected:
        part = archive.with_name(archive.name + ".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "4",
                "--continue-at",
                "-",
                "--output",
                str(part),
                item["url"],
            ],
            check=True,
        )
        actual = digest(part)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {key}: {actual}")
        part.rename(archive)
    if key != "qwen":
        target = path(".tools/" + key)
        target.mkdir(exist_ok=True)
        with tarfile.open(archive) as stream:
            stream.extractall(target, filter="data")
    else:
        write_json("models/qwen/provenance.json", item)
    print(
        json.dumps({"component": key, "sha256": expected, "bytes": archive.stat().st_size}),
        flush=True,
    )


if __name__ == "__main__":
    for name in sys.argv[1:]:
        fetch(name)
