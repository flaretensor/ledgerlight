import os

import numpy as np

from ledgerlight.onnx_backend import (
    OnnxCrossEncoder,
    OnnxEncoder,
    graph_path,
    tokenizer_for,
)


class OpenVinoSession:
    def __init__(self, filename):
        import openvino as ov

        device = os.environ.get("LEDGERLIGHT_OPENVINO_DEVICE", "CPU")
        config = {"INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "1"}
        if device == "CPU":
            config["INFERENCE_NUM_THREADS"] = 8
        self.model = ov.Core().compile_model(str(filename), device, config)

    def run(self, outputs, inputs):
        return [self.model(inputs)[0]]


class TritonSession:
    def __init__(self, role, version):
        import tritonclient.http as triton

        self.api = triton
        self.client = triton.InferenceServerClient(
            os.environ.get("LEDGERLIGHT_TRITON_URL", "127.0.0.1:8002"),
            connection_timeout=5,
            network_timeout=120,
        )
        self.name = role + "_" + str(version)
        if not self.client.is_model_ready(self.name):
            raise RuntimeError("Triton model unavailable: " + self.name)

    def run(self, outputs, inputs):
        tensors = []
        for name, array in inputs.items():
            tensor = self.api.InferInput(name, array.shape, "INT64")
            tensor.set_data_from_numpy(np.ascontiguousarray(array))
            tensors.append(tensor)
        result = self.client.infer(
            self.name, tensors, outputs=[self.api.InferRequestedOutput("output")]
        )
        return [result.as_numpy("output")]


class RuntimeEncoder(OnnxEncoder):
    def __init__(self, source, version, engine):
        self.tokenizer = tokenizer_for("encoder", source)
        self.runtime = (
            TritonSession("encoder", version)
            if os.environ.get("LEDGERLIGHT_TRITON", "0") == "1"
            else OpenVinoSession(graph_path("encoder", version))
        )
        self.provider = engine


class RuntimeCrossEncoder(OnnxCrossEncoder):
    def __init__(self, source, version, engine):
        self.tokenizer = tokenizer_for("cross_encoder", source)
        self.runtime = (
            TritonSession("cross_encoder", version)
            if os.environ.get("LEDGERLIGHT_TRITON", "0") == "1"
            else OpenVinoSession(graph_path("cross_encoder", version))
        )
        self.provider = engine
