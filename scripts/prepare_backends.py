import argparse
import gc
import json
import os

from ledgerlight.common import digest, path, read_json, write_json
from ledgerlight.onnx_backend import CrossExport, EncoderExport, export, graph_path
from ledgerlight.retrieval import load_model


def prepare_graphs():
    selected = read_json("models/production/config.json")
    records = []
    for role, wrapper in [("encoder", EncoderExport), ("cross_encoder", CrossExport)]:
        for version in ["1", "2"]:
            source = path(selected[role + "_path"]) if version == "2" else None
            target = graph_path(role, version)
            if not target.exists():
                model = load_model(role, source, device="cpu")
                export(wrapper(model), model.tokenizer, target)
                del model
                gc.collect()
            records.append({"role": role, "version": version, "sha256": digest(target)})
    write_json("data/runtime/onnx/manifest.json", records)
    return records


def prepare_triton(engine):
    repository = path("data/runtime/triton/" + engine)
    for role in ["encoder", "cross_encoder"]:
        for version in ["1", "2"]:
            name = role + "_" + version
            root = repository / name
            (root / "1").mkdir(parents=True, exist_ok=True)
            target = root / "1/model.onnx"
            if not target.exists():
                target.symlink_to(os.path.relpath(graph_path(role, version), target.parent))
            output = "[ 384 ]" if role == "encoder" else "[ 1 ] reshape { shape: [ ] }"
            if engine == "ort":
                backend = 'backend: "onnxruntime"'
                placement = "instance_group [ { kind: KIND_GPU count: 1 } ]"
                options = """optimization { execution_accelerators { gpu_execution_accelerator [ {
name: "cuda"
parameters { key: "use_tf32" value: "0" }
parameters { key: "gpu_mem_limit" value: "1073741824" }
parameters { key: "arena_extend_strategy" value: "kSameAsRequested" }
} ] } }
parameters { key: "intra_op_thread_count" value { string_value: "8" } }
parameters { key: "inter_op_thread_count" value { string_value: "1" } }"""
            else:
                backend = 'backend: "openvino"\ndefault_model_filename: "model.onnx"'
                placement = "instance_group [ { kind: KIND_CPU count: 1 } ]"
                options = """parameters { key: "INFERENCE_PRECISION_HINT" value { string_value: "f32" } }
parameters { key: "INFERENCE_NUM_THREADS" value { string_value: "8" } }
parameters { key: "NUM_STREAMS" value { string_value: "1" } }
parameters { key: "ENABLE_BATCH_PADDING" value { string_value: "YES" } }"""
            config = f'''name: "{name}"
{backend}
max_batch_size: 4
input [
{{ name: "input_ids" data_type: TYPE_INT64 dims: [ -1 ] }},
{{ name: "attention_mask" data_type: TYPE_INT64 dims: [ -1 ] }},
{{ name: "token_type_ids" data_type: TYPE_INT64 dims: [ -1 ] }}
]
output [ {{ name: "output" data_type: TYPE_FP32 dims: {output} }} ]
{placement}
{options}
'''
            (root / "config.pbtxt").write_text(config)
    return str(repository)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton", choices=["ort", "openvino"])
    args = parser.parse_args()
    result = {"graphs": prepare_graphs()}
    if args.triton:
        result["repository"] = prepare_triton(args.triton)
    print(json.dumps(result, indent=2))
