import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from ledgerlight.common import ROOT, digest, params, path, read_json, write_json


def build_bundle(destination):
    destination.mkdir(parents=True, exist_ok=True)
    selected = read_json("models/production/config.json")
    for key in ["encoder_path", "cross_encoder_path", "ltr_path"]:
        relative = selected[key]
        source = path(relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, target)
    for role in ["encoder", "cross_encoder"]:
        for version in ["1", "2"]:
            relative = f"data/runtime/onnx/{role}/{version}/model.onnx"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path(relative), target)
    frozen = read_json("data/evaluation/freeze.json")
    metrics = read_json("data/evaluation/metrics.json")
    cfg = params()
    provenance = {
        "format": 1,
        "training_fingerprint": frozen["training_fingerprint"],
        "dataset": {
            **cfg["data"],
            "name": "FinRank",
            "repository": "https://github.com/datanxt/FinRank",
            "license_url": "https://github.com/datanxt/FinRank/blob/"
            + cfg["data"]["revision"]
            + "/LICENSE",
            "splits_sha256": digest(path("data/raw/splits.json")),
            "annotations_license": "CC-BY-NC-4.0",
        },
        "base_models": {
            "encoder": {**cfg["encoder"], "license": "MIT"},
            "cross_encoder": {**cfg["cross_encoder"], "license": "Apache-2.0"},
        },
        "selection": {k: v for k, v in frozen.items() if k not in {"configurations", "dev"}},
        "metrics": {
            "dev": metrics["dev"]["CatBoost"],
            "test": metrics["test"]["CatBoost"],
            "paired_test_catboost_minus_bm25": metrics["bootstrap"],
        },
        "corpus_sha256": selected["corpus_sha256"],
        "selected_paths": {
            k: selected[k] for k in ["encoder_path", "cross_encoder_path", "ltr_path"]
        },
        "query_counts": {"train": 948, "dev": 118, "test": 119},
        "window_count": 12604,
        "metadata_license": "cc-by-nc-4.0",
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (destination / "pipelines.json").write_text(
        json.dumps(
            {
                "configurations": frozen["configurations"],
                "dev": frozen["dev"],
                "selected": frozen["selected"],
            },
            indent=2,
        )
        + "\n"
    )
    (destination / "README.md").write_text(
        "---\nlicense: cc-by-nc-4.0\nlanguage: en\nbase_model:\n- BAAI/bge-small-en-v1.5\n- cross-encoder/ms-marco-MiniLM-L6-v2\ntags:\n- information-retrieval\n- finance\n---\n"
    )
    records = []
    for item in sorted(destination.rglob("*")):
        if item.is_file() and item.name != "SHA256SUMS":
            records.append(digest(item) + "  " + str(item.relative_to(destination)))
    (destination / "SHA256SUMS").write_text("\n".join(records) + "\n")
    result = {
        "directory": str(destination.relative_to(ROOT)),
        "files": sum(p.is_file() for p in destination.rglob("*")),
        "bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
        "sums_sha256": digest(destination / "SHA256SUMS"),
    }
    write_json("data/analysis/serving-benchmarks/model-distribution/bundle.json", result)
    print(json.dumps(result, indent=2))


def verify_bundle(directory):
    records = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe bundle path")
        source = (directory / name).resolve()
        if (
            not source.is_relative_to(directory.resolve())
            or len(expected) != 64
            or digest(source) != expected
        ):
            raise ValueError("Bundle checksum or path violation: " + name)
        records[name] = expected
    if not {"provenance.json", "pipelines.json", "README.md"} <= set(records):
        raise ValueError("Incomplete model bundle")
    return records


def copy_verified(source, target, expected):
    if digest(source) != expected:
        raise ValueError("Cached source checksum mismatch: " + source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError("Existing target checksum mismatch: " + str(target.relative_to(ROOT)))
    else:
        shutil.copyfile(source, target)
    if digest(target) != expected:
        raise ValueError("Target copy checksum mismatch")


def cached_file(relative, url, expected, cache_root, reused):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    source = cache_root / relative if cache_root else None
    if source and source.is_file():
        copy_verified(source, target, expected)
        reused.append({"path": relative, "sha256": expected})
    elif not target.is_file():
        urllib.request.urlretrieve(url, target)
    if digest(target) != expected:
        raise ValueError("Pinned download checksum mismatch: " + relative)


def base_snapshots(cache_root, reused):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cfg = params()
    result = {}
    for role in ["encoder", "cross_encoder"]:
        specification = cfg[role]
        source = None
        if cache_root:
            try:
                source = Path(
                    snapshot_download(
                        specification["name"],
                        revision=specification["revision"],
                        cache_dir=cache_root / ".cache/huggingface/hub",
                        local_files_only=True,
                    )
                )
            except (FileNotFoundError, LocalEntryNotFoundError):
                pass
        if source:
            target = (
                ROOT
                / ".cache/huggingface/hub"
                / ("models--" + specification["name"].replace("/", "--"))
                / "snapshots"
                / specification["revision"]
            )
            for item in source.rglob("*"):
                if not item.is_file():
                    continue
                expected = digest(item)
                if item.is_symlink():
                    address = item.resolve().name
                    if len(address) == 64 and address != expected:
                        raise ValueError("HF content-address mismatch")
                    if len(address) == 40:
                        data = item.read_bytes()
                        git_hash = hashlib.sha1(
                            b"blob " + str(len(data)).encode() + b"\0" + data
                        ).hexdigest()
                        if address != git_hash:
                            raise ValueError("HF Git object mismatch")
                copy_verified(item, target / item.relative_to(source), expected)
                reused.append(
                    {
                        "base_model": specification["name"],
                        "revision": specification["revision"],
                        "file": str(item.relative_to(source)),
                        "sha256": expected,
                    }
                )
        else:
            target = Path(
                snapshot_download(
                    specification["name"],
                    revision=specification["revision"],
                    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "1_Pooling/**"],
                )
            )
        result[role] = target
    return result


def register_fresh(bundle, snapshots):
    import mlflow

    from ledgerlight.registry import (
        CrossEncoderAdapter,
        EncoderAdapter,
        PipelineAdapter,
        RankerAdapter,
        client,
        namespace,
        promote,
    )

    c = client()
    if list(c.search_registered_models()):
        raise ValueError("Bootstrap requires an empty isolated registry")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(namespace())
    provenance = json.loads((bundle / "provenance.json").read_text())
    configurations = json.loads((bundle / "pipelines.json").read_text())
    components = {}
    for role, name, adapter in [
        ("encoder", "encoder", EncoderAdapter),
        ("cross_encoder", "cross-encoder", CrossEncoderAdapter),
    ]:
        versions = []
        for kind, source in [
            ("base", snapshots[role]),
            ("trained", path(provenance["selected_paths"][role + "_path"])),
        ]:
            with mlflow.start_run(run_name="bootstrap-" + name + "-" + kind):
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=adapter(),
                    artifacts={"weights": str(source)},
                    registered_model_name=namespace() + "-" + name,
                    code_paths=["src/ledgerlight"],
                    pip_requirements=["sentence-transformers==5.2.3", "torch==2.8.0"],
                )
            versions.append(str(info.registered_model_version))
        components[name] = versions
        c.set_registered_model_alias(namespace() + "-" + name, "champion", versions[-1])
    with mlflow.start_run(run_name="bootstrap-ranker"):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=RankerAdapter(),
            artifacts={"weights": str(path(provenance["selected_paths"]["ltr_path"]))},
            registered_model_name=namespace() + "-ranker",
            code_paths=["src/ledgerlight"],
            pip_requirements=["catboost>=1.2.8,<1.3"],
        )
    components["ranker"] = str(info.registered_model_version)
    c.set_registered_model_alias(namespace() + "-ranker", "champion", components["ranker"])
    versions = {}
    for method, config in configurations["configurations"].items():
        selected = {
            **config,
            "components": {
                "encoder": components["encoder"][int(config["index"] == "trained")],
                "cross_encoder": components["cross-encoder"][
                    int(config["cross_encoder_path"] is not None)
                ]
                if config["mode"] in {"ce", "ltr"}
                else None,
                "ranker": components["ranker"] if config["mode"] == "ltr" else None,
            },
        }
        location = "data/registry/" + method + "/manifest.json"
        write_json(location, selected)
        with mlflow.start_run(run_name="bootstrap-pipeline-" + method):
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=PipelineAdapter(),
                artifacts={"manifest": str(path(location))},
                registered_model_name=namespace() + "-pipeline",
                code_paths=["src/ledgerlight"],
                pip_requirements=[
                    "sentence-transformers==5.2.3",
                    "torch==2.8.0",
                    "catboost",
                    "bm25s",
                    "qdrant-client",
                    "numpy",
                    "pyyaml",
                    "python-dotenv",
                    "mlflow",
                ],
            )
        version = str(info.registered_model_version)
        versions[method] = version
        c.set_model_version_tag(
            namespace() + "-pipeline",
            version,
            "dev_ndcg10",
            str(configurations["dev"][method]["ndcg@10"]),
        )
        c.set_model_version_tag(namespace() + "-pipeline", version, "method", method)
    c.set_registered_model_alias(namespace() + "-pipeline", "champion", versions["BM25"])
    promote(versions[configurations["selected"]])
    write_json(
        "data/registry/versions.json",
        {
            "pipelines": versions,
            "components": components,
            "selected": versions[configurations["selected"]],
        },
    )


def bootstrap(args):
    if path("data/evaluation/test.started.json").exists():
        raise PermissionError(
            "Bootstrap must run in a fresh checkout, never the evaluated canonical root"
        )
    started = time.monotonic()
    reused = []
    if args.repo:
        if not args.revision or not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
            raise ValueError("Hugging Face revision must be an immutable 40-character commit")
        from huggingface_hub import snapshot_download

        bundle = Path(
            snapshot_download(
                args.repo, revision=args.revision, local_dir=path("data/bootstrap/bundle")
            )
        )
    else:
        bundle = args.bundle.resolve()
    records = verify_bundle(bundle)
    provenance = json.loads((bundle / "provenance.json").read_text())
    cfg = params()
    if (
        provenance["dataset"]["sha256"] != cfg["data"]["sha256"]
        or provenance["dataset"]["revision"] != cfg["data"]["revision"]
    ):
        raise ValueError("Source and bundle pins differ")
    for name, expected in records.items():
        if name.startswith(("models/", "data/runtime/onnx/")):
            copy_verified(bundle / name, ROOT / name, expected)
    cached_file(
        "data/raw/FinRank.jsonl",
        f"https://raw.githubusercontent.com/datanxt/FinRank/{cfg['data']['revision']}/FinRank.jsonl",
        cfg["data"]["sha256"],
        args.cache_root,
        reused,
    )
    cached_file(
        "data/raw/splits.json",
        f"https://raw.githubusercontent.com/datanxt/FinRank/{cfg['data']['revision']}/baselines/splits.json",
        provenance["dataset"]["splits_sha256"],
        args.cache_root,
        reused,
    )
    qwen = read_json("services.lock.json")["qwen"]
    cached_file(
        "models/qwen/" + qwen["name"],
        qwen["url"],
        qwen["digest"].split(":")[-1],
        args.cache_root,
        reused,
    )
    snapshots = base_snapshots(args.cache_root, reused)
    llama = read_json("services.lock.json")["llama"]
    cached_file(
        ".tools/downloads/" + llama["name"],
        llama["url"],
        llama["digest"].split(":")[-1],
        args.cache_root,
        reused,
    )
    path(".tools/llama").mkdir(parents=True, exist_ok=True)
    with tarfile.open(path(".tools/downloads/" + llama["name"])) as archive:
        archive.extractall(path(".tools/llama"), filter="data")
    from ledgerlight.data import ingest, validate
    from ledgerlight.retrieval import embed, lexical_index

    ingest()
    validate()
    lexical_index()
    embed()
    embed(path(provenance["selected_paths"]["encoder_path"]), "indices/trained")
    if digest(path("data/processed/corpus.jsonl")) != provenance["corpus_sha256"]:
        raise ValueError("Rebuilt corpus does not match bundle")
    env_file = ROOT / ".env"
    if not env_file.exists():
        env_file.write_text(
            "POSTGRES_USER=ledgerlight\nPOSTGRES_DB=ledgerlight\nPOSTGRES_PORT=5433\nPOSTGRES_PASSWORD="
            + secrets.token_hex(24)
            + "\nGRAFANA_PASSWORD="
            + secrets.token_hex(24)
            + "\n"
        )
        env_file.chmod(0o600)
    from dotenv import load_dotenv

    load_dotenv(env_file, override=True)
    for directory in [
        "data/postgres",
        "mlruns/artifacts",
        "data/runtime/serving",
        "data/qdrant/storage",
        "data/qdrant/snapshots",
        "data/prometheus",
        "data/grafana",
        ".cache/registry",
    ]:
        path(directory).mkdir(parents=True, exist_ok=True)
    from prepare_backends import prepare_triton

    for engine in ["ort", "openvino"]:
        prepare_triton(engine)
    import yaml

    override = path("data/runtime/bootstrap.compose.yaml")
    override.parent.mkdir(parents=True, exist_ok=True)
    configuration = {"services": {"mlflow": {"volumes": ["./mlruns:/workspace/mlruns"]}}}
    if args.cache_root:
        configuration["services"]["retrieval"] = {
            "build": {
                "additional_contexts": {
                    "dependency_cache": str(args.cache_root / ".cache/uv"),
                    "python_runtime": str(args.cache_root / ".python"),
                }
            }
        }
    override.write_text(yaml.safe_dump(configuration))
    project = os.environ.get("LEDGERLIGHT_COMPOSE_PROJECT", "ledgerlight")
    command = [
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "-p",
        project,
        "-f",
        str(ROOT / "compose.yaml"),
        "-f",
        str(override),
    ]
    subprocess.run(command + ["build", "retrieval", "forwarder"], check=True)
    subprocess.run(
        command + ["up", "-d", "--no-build", "--pull", "never", "postgres", "mlflow"], check=True
    )
    import httpx

    for _ in range(120):
        try:
            httpx.get("http://127.0.0.1:5000/health", timeout=3).raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(1)
    register_fresh(bundle, snapshots)
    result = {
        "seconds": time.monotonic() - started,
        "reused": reused,
        "bundle_sums_sha256": digest(bundle / "SHA256SUMS"),
        "registered": read_json("data/registry/versions.json"),
    }
    write_json("data/bootstrap/receipt.json", result)
    if args.prepare_only:
        print(json.dumps({"seconds": result["seconds"], "reused_files": len(reused)}, indent=2))
        return
    os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/site.py"), "start"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("bundle")
    build.add_argument("--output", type=Path, default=ROOT / "data/release/ledgerlight-models")
    boot = commands.add_parser("bootstrap")
    choice = boot.add_mutually_exclusive_group(required=True)
    choice.add_argument("--bundle", type=Path)
    choice.add_argument("--repo")
    boot.add_argument("--revision")
    boot.add_argument("--cache-root", type=Path)
    boot.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.action == "bundle":
        build_bundle(args.output)
    else:
        bootstrap(args)
