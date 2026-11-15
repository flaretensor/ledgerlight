import os
import threading

import numpy as np

from ledgerlight.onnx_backend import (
    OnnxCrossEncoder,
    OnnxEncoder,
    graph_path,
    tokenizer_for,
)


def synchronize_triton(configuration):
    import tritonclient.http as triton

    selected = configuration["components"]
    wanted = {"encoder_" + str(selected["encoder"])}
    if configuration["mode"] in {"ce", "ltr"}:
        wanted.add("cross_encoder_" + str(selected["cross_encoder"]))
    server = triton.InferenceServerClient(
        os.environ.get("LEDGERLIGHT_TRITON_URL", "127.0.0.1:8002"), network_timeout=180
    )
    try:
        for item in server.get_model_repository_index():
            name = item["name"]
            if (
                name.startswith(("encoder_", "cross_encoder_"))
                and item.get("state") == "READY"
                and name not in wanted
            ):
                server.unload_model(name)
        for name in sorted(wanted):
            if not server.is_model_ready(name):
                server.load_model(name)
    finally:
        server.close()


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
    def __init__(self, role, version, engine):
        import tritonclient.http as triton

        self.api = triton
        self.local = threading.local()
        self.name = role + "_" + str(version)
        self.pad_batch = engine.endswith("openvino")

    def client(self):
        if not hasattr(self.local, "client"):
            self.local.client = self.api.InferenceServerClient(
                os.environ.get("LEDGERLIGHT_TRITON_URL", "127.0.0.1:8002"),
                connection_timeout=5,
                network_timeout=120,
            )
            if not self.local.client.is_model_ready(self.name):
                raise RuntimeError("Triton model unavailable: " + self.name)
        return self.local.client

    def run(self, outputs, inputs):
        size = len(next(iter(inputs.values())))
        if size > 4:
            return [
                np.concatenate(
                    [
                        self.run(outputs, {k: v[i : i + 4] for k, v in inputs.items()})[0]
                        for i in range(0, size, 4)
                    ]
                )
            ]
        if self.pad_batch and size < 4:
            # Avoid the backend's partial-batch buffer path for dynamic ONNX inputs.
            # Rows are independent; discard the duplicate rows after inference.
            inputs = {
                name: np.concatenate([array, np.repeat(array[-1:], 4 - size, axis=0)])
                for name, array in inputs.items()
            }
        tensors = []
        for name, array in inputs.items():
            tensor = self.api.InferInput(name, array.shape, "INT64")
            tensor.set_data_from_numpy(np.ascontiguousarray(array))
            tensors.append(tensor)
        result = self.client().infer(
            self.name, tensors, outputs=[self.api.InferRequestedOutput("output")]
        )
        return [result.as_numpy("output")[:size]]


class RuntimeEncoder(OnnxEncoder):
    def __init__(self, source, version, engine):
        self.tokenizer = tokenizer_for("encoder", source)
        self.runtime = (
            TritonSession("encoder", version, engine)
            if os.environ.get("LEDGERLIGHT_TRITON", "0") == "1"
            else OpenVinoSession(graph_path("encoder", version))
        )
        self.provider = engine


class RuntimeCrossEncoder(OnnxCrossEncoder):
    def __init__(self, source, version, engine):
        self.tokenizer = tokenizer_for("cross_encoder", source)
        self.runtime = (
            TritonSession("cross_encoder", version, engine)
            if os.environ.get("LEDGERLIGHT_TRITON", "0") == "1"
            else OpenVinoSession(graph_path("cross_encoder", version))
        )
        self.provider = engine
