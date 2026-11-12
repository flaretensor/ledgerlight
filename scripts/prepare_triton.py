import json
import os

from ledgerlight.common import path, write_json

for name, dims in [("encoder", "[ 384 ]"), ("cross_encoder", "[ 1 ]")]:
    root = path("models/triton/" + name)
    (root / "1").mkdir(parents=True, exist_ok=True)
    target = root / "1/model.onnx"
    if not target.exists() and not target.is_symlink():
        os.symlink("../../../onnx/" + name + "/model.onnx", target)
    reshape = " reshape { shape: [ ] }" if name == "cross_encoder" else ""
    config = f'''name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
 {{ name: "input_ids" data_type: TYPE_INT64 dims: [ -1 ] }},
 {{ name: "attention_mask" data_type: TYPE_INT64 dims: [ -1 ] }},
 {{ name: "token_type_ids" data_type: TYPE_INT64 dims: [ -1 ] }}
]
output [ {{ name: "output" data_type: TYPE_FP32 dims: {dims}{reshape} }} ]
dynamic_batching {{ preferred_batch_size: [ 4, 8 ] max_queue_delay_microseconds: 2000 }}
instance_group [ {{ kind: KIND_GPU count: 1 }} ]
optimization {{ execution_accelerators {{ gpu_execution_accelerator [ {{
 name: "cuda"
 parameters {{ key: "use_tf32" value: "0" }}
 parameters {{ key: "gpu_mem_limit" value: "1073741824" }}
 parameters {{ key: "arena_extend_strategy" value: "kSameAsRequested" }}
}} ] }} }}
parameters {{ key: "intra_op_thread_count" value {{ string_value: "8" }} }}
parameters {{ key: "inter_op_thread_count" value {{ string_value: "1" }} }}
'''
    (root / "config.pbtxt").write_text(config)
write_json(
    "data/serving/triton_preparation.json",
    {
        "model_repository": "models/triton",
        "backend": "onnxruntime",
        "max_batch_size": 8,
        "validated_runtime": False,
    },
)
print(json.dumps({"prepared": "models/triton", "runtime_validated": False}))
