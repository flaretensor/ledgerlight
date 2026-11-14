import os

if os.environ.get("LEDGERLIGHT_RUN_MODE") != "pilot":
    raise SystemExit("Set LEDGERLIGHT_RUN_MODE=pilot")

import mlflow

from ledgerlight.common import digest, path, read_json, write_json
from ledgerlight.final_evaluation import configuration
from ledgerlight.registry import PIPELINE_NAME, PipelineAdapter, RankerAdapter, client

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ledgerlight-pilot")
c = client()
encoder = read_json("data/training/encoder_selected.json")
cross = read_json("data/training/ce_selected.json")
ltr = read_json("data/training/ltr_selected.json")
components = {}
for key, name in [("encoder", "encoder"), ("cross_encoder", "cross-encoder")]:
    versions = c.search_model_versions(f"name='ledgerlight-pilot-{name}'")
    if len(versions) != 1:
        raise RuntimeError("Expected one unchanged pilot neural component")
    components[key] = str(versions[0].version)
with mlflow.start_run(run_name="pilot-ticker-register-ranker"):
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=RankerAdapter(),
        artifacts={"weights": str(path(ltr["path"]))},
        code_paths=["src/ledgerlight"],
        registered_model_name="ledgerlight-pilot-ranker",
        pip_requirements=["catboost==1.2.10"],
    )
components["ranker"] = str(info.registered_model_version)
config = configuration("CatBoost", ltr, encoder, cross, read_json("data/baselines/metrics.json"))
config.update(
    components=components, pilot=True, feature_sha256=digest(path("src/ledgerlight/features.py"))
)
if config["index"] != "trained":
    raise RuntimeError("This smoke expects the existing trained pilot encoder")
write_json("data/registry/pilot/manifest.json", config)
with mlflow.start_run(run_name="pilot-ticker-register-pipeline"):
    mlflow.log_metric("dev_ndcg10", ltr["dev_ndcg_mean"])
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=PipelineAdapter(),
        artifacts={"manifest": str(path("data/registry/pilot/manifest.json"))},
        code_paths=["src/ledgerlight"],
        registered_model_name=PIPELINE_NAME,
        pip_requirements=["mlflow==" + mlflow.__version__],
    )
version = str(info.registered_model_version)
for key, name in [("encoder", "encoder"), ("cross_encoder", "cross-encoder"), ("ranker", "ranker")]:
    c.set_registered_model_alias("ledgerlight-pilot-" + name, "champion", components[key])
c.set_registered_model_alias(PIPELINE_NAME, "champion", version)
write_json(
    "data/registry/versions.json",
    {
        "selected": version,
        "components": {
            "encoder": [components["encoder"]],
            **{k: v for k, v in components.items() if k != "encoder"},
        },
    },
)
print({"pipeline": PIPELINE_NAME, "version": version, "components": components})
