import json
import os
import shutil

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from ledgerlight.common import namespace, path, read_json, write_json

PIPELINE_NAME = namespace() + "-pipeline"


class EncoderAdapter(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(context.artifacts["weights"], device="cpu")

    def predict(self, context, model_input, params=None):
        return self.model.encode(model_input["text"].tolist(), normalize_embeddings=True)


class CrossEncoderAdapter(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(context.artifacts["weights"], device="cpu", max_length=512)

    def predict(self, context, model_input, params=None):

        pairs = list(zip(model_input["question"], model_input["text"]))
        from ledgerlight.pairs import torch_pair_scores

        return torch_pair_scores(self.model, pairs)


class RankerAdapter(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from catboost import CatBoostRanker

        self.model = CatBoostRanker()
        self.model.load_model(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        return self.model.predict(model_input)


class PipelineAdapter(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ledgerlight.engine import SearchEngine

        self.engine = SearchEngine(
            resolve_components(json.loads(open(context.artifacts["manifest"]).read()))
        )

    def predict(self, context, model_input, params=None):
        return [self.engine.search(question) for question in model_input["question"]]


def client():
    return MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))


def acceptable(candidate, champion):
    import math

    return math.isfinite(candidate) and (champion is None or candidate >= champion)


def resolve_components(config):
    import yaml

    result = dict(config)
    selected = config.get("components", {})
    for key, model_name, output_key in [
        ("encoder", "ledgerlight-encoder", "encoder_path"),
        ("cross_encoder", "ledgerlight-cross-encoder", "cross_encoder_path"),
        ("ranker", "ledgerlight-ranker", "ltr_path"),
    ]:
        model_name = model_name.replace("ledgerlight-", namespace() + "-", 1)
        version = selected.get(key)
        if version is None or key == "cross_encoder" and config["mode"] not in {"ce", "ltr"}:
            continue
        destination = path(f".cache/registry/components/{model_name}/{version}")
        destination.mkdir(parents=True, exist_ok=True)
        root = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{version}", dst_path=str(destination)
        )
        metadata = yaml.safe_load((path(root) / "MLmodel").read_text())
        relative = metadata["flavors"]["python_function"]["artifacts"]["weights"]["path"]
        result[output_key] = str(path(root) / relative)
    return result


def manifest(version=None, alias="champion"):
    c = client()
    model = (
        c.get_model_version(PIPELINE_NAME, str(version))
        if version is not None
        else c.get_model_version_by_alias(PIPELINE_NAME, alias)
    )
    uri = f"models:/{PIPELINE_NAME}/{model.version}"
    target = mlflow.artifacts.download_artifacts(
        artifact_uri=uri + "/artifacts/manifest.json", dst_path=str(path(".cache/registry"))
    )
    result = json.loads(open(target).read())
    result["registry_version"] = str(model.version)
    return resolve_components(result)


def component_aliases(configuration):
    c = client()
    for key, name in [
        ("encoder", "ledgerlight-encoder"),
        ("cross_encoder", "ledgerlight-cross-encoder"),
        ("ranker", "ledgerlight-ranker"),
    ]:
        name = name.replace("ledgerlight-", namespace() + "-", 1)
        version = configuration.get("components", {}).get(key)
        if version is None:
            continue
        try:
            previous = c.get_model_version_by_alias(name, "champion")
        except mlflow.MlflowException:
            previous = None
        if previous is not None and str(previous.version) != str(version):
            c.set_registered_model_alias(name, "previous", previous.version)
        c.set_registered_model_alias(name, "champion", str(version))


def promote(version):
    c = client()
    candidate = c.get_model_version(PIPELINE_NAME, str(version))
    metric = float(candidate.tags["dev_ndcg10"])
    try:
        champion = c.get_model_version_by_alias(PIPELINE_NAME, "champion")
    except mlflow.MlflowException:
        champion = None
    current = float(champion.tags["dev_ndcg10"]) if champion is not None else None
    if not acceptable(metric, current):
        raise ValueError(f"Gate rejected candidate DEV nDCG@10 {metric} < {current}")
    if champion is not None and str(champion.version) != str(version):
        c.set_registered_model_alias(PIPELINE_NAME, "previous", champion.version)
    component_aliases(manifest(version))
    c.set_registered_model_alias(PIPELINE_NAME, "champion", str(version))
    result = {
        "action": "promote",
        "version": str(version),
        "dev_ndcg10": metric,
        "previous": str(champion.version) if champion is not None else None,
    }
    print(json.dumps(result))
    return result


def rollback():
    c = client()
    previous = c.get_model_version_by_alias(PIPELINE_NAME, "previous")
    current = c.get_model_version_by_alias(PIPELINE_NAME, "champion")
    component_aliases(manifest(previous.version))
    c.set_registered_model_alias(PIPELINE_NAME, "champion", previous.version)
    c.set_registered_model_alias(PIPELINE_NAME, "previous", current.version)
    result = {
        "action": "rollback",
        "version": str(previous.version),
        "previous": str(current.version),
    }
    print(json.dumps(result))
    return result


def register_all():
    from huggingface_hub import snapshot_download

    from ledgerlight.common import params

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(namespace())
    c = client()
    components = {}
    for key, config_key, selected_file, adapter in [
        ("encoder", "encoder", "encoder_selected.json", EncoderAdapter),
        ("cross-encoder", "cross_encoder", "ce_selected.json", CrossEncoderAdapter),
    ]:
        cfg = params()[config_key]
        selected = read_json("data/training/" + selected_file)
        snapshot = snapshot_download(cfg["name"], revision=cfg["revision"], local_files_only=True)
        materialized = path(f".cache/registry-input/{key}/{cfg['revision']}")
        shutil.copytree(snapshot, materialized, symlinks=False, dirs_exist_ok=True)
        weights = [
            str(materialized),
            str(path(selected["path"])),
        ]
        components[key] = []
        for kind, source in zip(["base", "trained"], weights):
            with mlflow.start_run(run_name=f"register-{key}-{kind}"):
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=adapter(),
                    artifacts={"weights": source},
                    registered_model_name=f"{namespace()}-{key}",
                    code_paths=["src/ledgerlight"],
                    pip_requirements=["sentence-transformers==5.2.3", "torch==2.8.0"],
                )
            components[key].append(str(info.registered_model_version))
        c.set_registered_model_alias(f"{namespace()}-{key}", "champion", components[key][-1])
    selected_ltr = read_json("data/training/ltr_selected.json")
    with mlflow.start_run(run_name="register-ranker"):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=RankerAdapter(),
            artifacts={"weights": str(path(selected_ltr["path"]))},
            registered_model_name=namespace() + "-ranker",
            code_paths=["src/ledgerlight"],
            pip_requirements=["catboost>=1.2.8,<1.3"],
        )
    components["ranker"] = str(info.registered_model_version)
    c.set_registered_model_alias(namespace() + "-ranker", "champion", components["ranker"])
    freeze = read_json("data/evaluation/freeze.json")
    versions = {}
    for method, configuration in freeze["configurations"].items():
        manifest_path = f"data/registry/{method}/manifest.json"
        write_json(
            manifest_path,
            {
                **configuration,
                "components": {
                    "encoder": components["encoder"][int(configuration["index"] == "trained")],
                    "cross_encoder": components["cross-encoder"][
                        int(configuration["cross_encoder_path"] is not None)
                    ]
                    if configuration["mode"] in {"ce", "ltr"}
                    else None,
                    "ranker": components["ranker"] if configuration["mode"] == "ltr" else None,
                },
            },
        )
        with mlflow.start_run(run_name=f"register-pipeline-{method}"):
            mlflow.log_metric("dev_ndcg10", freeze["dev"][method]["ndcg@10"])
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=PipelineAdapter(),
                artifacts={"manifest": str(path(manifest_path))},
                registered_model_name=PIPELINE_NAME,
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
        c.set_model_version_tag(
            PIPELINE_NAME, version, "dev_ndcg10", str(freeze["dev"][method]["ndcg@10"])
        )
        c.set_model_version_tag(PIPELINE_NAME, version, "method", method)
        versions[method] = version
    c.set_registered_model_alias(PIPELINE_NAME, "champion", versions["BM25"])
    promote(versions[freeze["selected"]])
    write_json(
        "data/registry/versions.json",
        {"pipelines": versions, "components": components, "selected": versions[freeze["selected"]]},
    )


if __name__ == "__main__":
    register_all()
