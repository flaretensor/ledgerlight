import argparse
import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

from runtime_helpers import OUT, save, summary

from ledgerlight.common import ROOT, digest


def run(command, directory, name, env=None):
    clock = time.monotonic()
    with (directory / (name + ".log")).open("w") as stream:
        subprocess.run(
            command,
            cwd=Path(env["LEDGERLIGHT_ROOT"]) if env else ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.monotonic() - clock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "clean_clone")
    directory = parser.parse_args().output.resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("Receipt directory escapes the project")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        raise RuntimeError("A completed clean clone must not be repeated")
    clone = ROOT / ".tmp/clean-clone/ledgerlight"
    if clone.exists() or clone.is_symlink():
        raise RuntimeError("Refusing to replace an existing clean checkout")
    if subprocess.check_output(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.docker.compose.project=ledgerlight-cleanclone",
        ],
        text=True,
    ).strip():
        raise RuntimeError("Refusing to reuse a pre-existing clean-clone Compose project")
    for resource in ["volume", "network"]:
        if subprocess.check_output(
            [
                "docker",
                resource,
                "ls",
                "-q",
                "--filter",
                "label=com.docker.compose.project=ledgerlight-cleanclone",
            ],
            text=True,
        ).strip():
            raise RuntimeError(
                "Refusing to reuse pre-existing clean-clone " + resource + " resources"
            )
    pending = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    if pending:
        raise RuntimeError("Commit and push the source before checking origin main")
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote_head = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True
    ).strip()
    if source_head != remote_head:
        raise RuntimeError("Commit and push the source before checking origin main")
    if not clone.parent.resolve().is_relative_to(ROOT):
        raise RuntimeError("Clean checkout parent escapes the project")
    clone.parent.mkdir(parents=True, exist_ok=True)
    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True
    ).strip()
    result = {
        "source_head": source_head,
        "origin_tracking_head": remote_head,
        "timings": {},
        "reused_images": [],
        "manual_steps": [
            "Obtain private repository access and configure SSH authentication before cloning.",
            "Install Python 3.12, uv, Docker Compose, and configure the GPU container runtime if using GPU inference.",
            "Clone the source, then source scripts/env.sh and run uv sync --frozen.",
            "Populate the pinned Compose service images (docker compose --profile triton pull --ignore-buildable) before offline startup.",
            "Run python scripts/release_models.py bootstrap --bundle PATH, or --repo REPOSITORY --revision IMMUTABLE_REVISION.",
            "Use python scripts/site.py start|stop|status; public mode requires separately configured tunnel credentials and DNS.",
        ],
    }
    run([str(ROOT / ".venv/bin/python"), "scripts/site.py", "stop"], directory, "stop-main")
    result["timings"]["clone"] = run(
        ["git", "clone", "--no-tags", "--single-branch", "--branch", "main", origin, str(clone)],
        directory,
        "clone",
    )
    result["clone_source"] = "origin main"
    result["clone_head"] = subprocess.check_output(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True
    ).strip()
    if result["clone_head"] != source_head:
        raise RuntimeError("Origin changed while preparing the clean clone")
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("POSTGRES_", "GRAFANA_")) or key in {
            "LEDGERLIGHT_EXPERIMENT",
            "LEDGERLIGHT_STUDY_SUFFIX",
            "LEDGERLIGHT_REQUEST_DEADLINE",
        }:
            env.pop(key)
    env.update(
        LEDGERLIGHT_ROOT=str(clone),
        LEDGERLIGHT_RUN_MODE="canonical",
        LEDGERLIGHT_COMPOSE_PROJECT="ledgerlight-cleanclone",
        HF_HOME=str(clone / ".cache/huggingface"),
        UV_CACHE_DIR=str(ROOT / ".cache/uv"),
        UV_PYTHON_INSTALL_DIR=str(ROOT / ".python"),
        XDG_CACHE_HOME=str(clone / ".cache"),
        XDG_CONFIG_HOME=str(clone / ".config"),
        TMPDIR=str(clone / ".tmp"),
        NUMBA_CACHE_DIR=str(clone / ".cache/numba"),
        CUDA_CACHE_PATH=str(clone / ".cache/cuda"),
        DVC_SITE_CACHE_DIR=str(clone / ".cache/dvc"),
        MLFLOW_TRACKING_URI="http://127.0.0.1:5000",
    )
    (clone / ".tmp").mkdir(parents=True, exist_ok=True)
    local = clone / ".env"
    local.write_text(
        "POSTGRES_USER=ledgerlight\nPOSTGRES_DB=ledgerlight\nPOSTGRES_PORT=5433\nPOSTGRES_PASSWORD="
        + secrets.token_hex(24)
        + "\nGRAFANA_PASSWORD="
        + secrets.token_hex(24)
        + "\n"
    )
    local.chmod(0o600)
    try:
        cache_hashes = {}
        for cache_directory in [ROOT / ".python", ROOT / ".cache/uv/archive-v0"]:
            for file in cache_directory.rglob("*"):
                if file.is_file() and not file.is_symlink():
                    cache_hashes[str(file.relative_to(ROOT))] = digest(file)
        save(directory / "reused-cache-sha256.json", cache_hashes)
        result["verified_cache_files"] = len(cache_hashes)
        result["timings"]["uv_sync"] = run(
            ["uv", "sync", "--frozen", "--link-mode", "copy"], directory, "uv-sync", env
        )
        result["timings"]["editable_import"] = run(
            [
                str(clone / ".venv/bin/python"),
                "-c",
                "from pathlib import Path; import ledgerlight; from ledgerlight.common import ROOT; assert Path(ledgerlight.__file__).is_relative_to(ROOT); print(ledgerlight.__file__)",
            ],
            directory,
            "editable-import",
            env,
        )
        for image in [
            "pgvector/pgvector:pg18",
            "qdrant/qdrant:v1.20.0",
            "prom/prometheus:v3.16.0",
            "grafana/grafana:13.3.1",
            "nvcr.io/nvidia/tritonserver:26.01-py3",
            "python:3.12.12-alpine3.22",
        ]:
            metadata = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
            result["reused_images"].append(
                {"image": image, "id": metadata["Id"], "digests": metadata["RepoDigests"]}
            )
        result["timings"]["bootstrap"] = run(
            [
                str(clone / ".venv/bin/python"),
                "scripts/release_models.py",
                "bootstrap",
                "--bundle",
                str(ROOT / "data/release/ledgerlight-models"),
                "--cache-root",
                str(ROOT),
                "--prepare-only",
            ],
            directory,
            "bootstrap",
            env,
        )
        save(
            directory / "bootstrap-receipt.json",
            json.loads((clone / "data/bootstrap/receipt.json").read_text()),
        )
        result["timings"]["start"] = run(
            [str(clone / ".venv/bin/python"), "scripts/site.py", "start"], directory, "start", env
        )
        program = """import json,httpx
from pathlib import Path
from ledgerlight.common import read_json,read_jsonl,digest
from ledgerlight.evaluation import evaluate
ids=set(read_json("data/processed/splits.json")["dev"])
queries=[q for q in read_jsonl("data/processed/queries.jsonl") if q["id"] in ids]
out={"ask":[],"rankings":{}}
with httpx.Client(timeout=300) as client:
 for q in queries[:3]:
  response=client.post("http://127.0.0.1:8000/ask",json={"question":q["question"]})
  out["ask"].append({"qid":q["id"],"http_status":response.status_code,"cited":bool(response.json().get("citations")),"abstained":response.json().get("abstained")})
 for q in queries:
  response=client.post("http://127.0.0.1:8001/search",json={"question":q["question"],"limit":20});response.raise_for_status()
  out["rankings"][q["id"]]=[r["passage_id"] for r in response.json()["results"]]
labels=read_json("data/processed/labels.json");out["metrics"],_=evaluate(out["rankings"],{q:labels[q] for q in out["rankings"]})
out["passed"]=all(r["http_status"]==200 for r in out["ask"]) and abs(out["metrics"]["ndcg@10"]-.4367276797012511)<=.005
out["rebuilt_sha256"]={p:digest(Path(p)) for p in ["data/processed/corpus.jsonl","indices/base/windows.jsonl","indices/base/vectors.npy","indices/trained/vectors.npy"]}
Path("data/bootstrap/dev-check.json").write_text(json.dumps(out,indent=2)+"\\n")
print(json.dumps({k:v for k,v in out.items() if k!="rankings"},indent=2))
assert out["passed"]
"""
        result["timings"]["checks"] = run(
            [str(clone / ".venv/bin/python"), "-c", program], directory, "checks", env
        )
        result["checks"] = json.loads((clone / "data/bootstrap/dev-check.json").read_text())
        save(directory / "dev-check.json", result["checks"])
        result["cache_files_changed"] = [
            name
            for name, expected in cache_hashes.items()
            if not (ROOT / name).is_file() or digest(ROOT / name) != expected
        ]
        result["passed"] = result["checks"]["passed"] and not result["cache_files_changed"]
    except Exception as exc:
        result["passed"] = False
        result["error"] = type(exc).__name__ + ": " + str(exc)
        save(directory / "failure.json", result)
        raise
    finally:
        if clone.exists():
            stop = (
                run(
                    [str(clone / ".venv/bin/python"), "scripts/site.py", "stop"],
                    directory,
                    "stop-clone",
                    env,
                )
                if (clone / ".venv/bin/python").exists()
                else None
            )
            result["timings"]["stop"] = stop
            if (clone / "data/runtime/stop.json").exists():
                result["stop"] = json.loads((clone / "data/runtime/stop.json").read_text())
            run(
                [
                    "docker",
                    "compose",
                    "--project-directory",
                    str(clone),
                    "-p",
                    "ledgerlight-cleanclone",
                    "-f",
                    str(clone / "compose.yaml"),
                    "--profile",
                    "triton",
                    "--profile",
                    "public",
                    "down",
                    "--volumes",
                    "--remove-orphans",
                ],
                directory,
                "remove-clone-resources",
                env,
            )
            shutil.rmtree(clone)
            result["clone_deleted"] = not clone.exists()
            if not any(clone.parent.iterdir()):
                clone.parent.rmdir()
        save(directory / "attempt.json", result)
    save(directory / "result.json", result)
    if directory == (OUT / "clean_clone").resolve():
        summary("release", "clean_clone", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
