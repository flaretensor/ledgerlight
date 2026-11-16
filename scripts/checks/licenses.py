import hashlib
import importlib.metadata as metadata
import json
import re
import tomllib
from pathlib import Path

from ledgerlight.common import write_json

configuration = tomllib.loads(Path("pyproject.toml").read_text())
requirements = configuration["project"]["dependencies"] + configuration["dependency-groups"]["dev"]
rows = []
for requirement in requirements:
    name = re.split(r"[\[<>=!~]", requirement)[0]
    distribution = metadata.distribution(name)
    record = distribution.metadata
    expression = record.get("License-Expression")
    declaration = (
        expression
        or record.get("License")
        or "; ".join(value for value in record.get_all("Classifier", []) if "License" in value)
    )
    bundled = []
    for file in distribution.files or []:
        if str(file).startswith(distribution._path.name) and any(
            token in str(file).lower() for token in ["license", "copying"]
        ):
            location = distribution.locate_file(file)
            text = location.read_text(errors="replace")
            bundled.append(
                {"file": str(file), "sha256": hashlib.sha256(location.read_bytes()).hexdigest()}
            )
            if "BSD 3-Clause License" in text[:200]:
                declaration = "BSD-3-Clause"
            elif text.lstrip().startswith("MIT License"):
                declaration = "MIT"
            elif "Apache License" in text[:200] and "Version 2.0" in text[:300]:
                declaration = "Apache-2.0"
    if declaration and "License ::" in declaration:
        declaration = declaration.replace("License :: OSI Approved :: ", "")
    if declaration in {
        "Apache",
        "Apache 2.0",
        "Apache 2.0 License",
        "Apache2.0",
        "Apache License, Version 2.0",
    }:
        declaration = "Apache-2.0"
    if declaration in {"MIT License"}:
        declaration = "MIT"
    if not declaration or len(declaration) > 250:
        declaration = "not verified"
    rows.append(
        {
            "component": name,
            "version": distribution.version,
            "declared_license": declaration,
            "source": f"https://pypi.org/project/{name}/{distribution.version}/",
            "verification": "Installed distribution's own versioned METADATA and bundled license files",
            "metadata_sha256": hashlib.sha256(
                distribution.read_text("METADATA").encode()
            ).hexdigest(),
            "bundled_license_files": bundled,
        }
    )
upstream = json.loads(Path(".cache/licenses/upstream-fetch.json").read_text())
qwen = json.loads(Path(".cache/licenses/qwen-upstream-audit.json").read_text())
write_json(
    "reports/licenses.json",
    {
        "checked_date": "2026-11-12",
        "python": rows,
        "pinned_upstream_fetches": upstream,
        "qwen_upstream": qwen,
        "triton": {
            "image": "nvcr.io/nvidia/tritonserver:26.01-py3",
            "server_license": "BSD-3-Clause",
            "server_license_sha256": hashlib.sha256(
                Path(".cache/licenses/triton-source-license.txt").read_bytes()
            ).hexdigest(),
            "source": "https://github.com/triton-inference-server/server/blob/r26.01/LICENSE",
            "container_and_all_bundled_terms": "not verified",
            "note": "Server license read from the retained image. The checked /NGC-DL-CONTAINER-LICENSE path was absent; no inference about full image licensing follows.",
        },
        "code_license": "not specified",
        "dataset_license": "CC-BY-NC-4.0",
        "interpretation": "License declarations only; not legal conclusions or an exhaustive transitive-license review",
    },
)
print(
    json.dumps(
        [{k: row[k] for k in ["component", "version", "declared_license"]} for row in rows],
        indent=2,
    )
)
