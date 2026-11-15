import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from ledgerlight.common import params, path


class EncoderExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model[0].auto_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).last_hidden_state[:, 0]
        return F.normalize(output, p=2, dim=1)


class CrossExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).logits.reshape(-1)


def export(wrapper, tokenizer, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    inputs = tokenizer(
        ["financial business revenue"] * 2,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    names = ["input_ids", "attention_mask", "token_type_ids"]
    wrapper.eval().cpu()
    torch.onnx.export(
        wrapper,
        tuple(inputs[k] for k in names),
        str(destination),
        input_names=names,
        output_names=["output"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={**{k: {0: "batch", 1: "sequence"} for k in names}, "output": {0: "batch"}},
    )


def session(filename, provider):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        ort.preload_dlls(directory="")
        providers = [
            (
                provider,
                {
                    "gpu_mem_limit": 1024**3,
                    "use_tf32": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        ]
    result = ort.InferenceSession(str(filename), sess_options=options, providers=providers)
    if result.get_providers()[0] != provider:
        raise RuntimeError(f"Requested {provider}; actual {result.get_providers()}")
    return result


def graph_path(role, version):
    return path(f"data/runtime/onnx/{role}/{version}/model.onnx")


def tokenizer_for(role, source):
    from transformers import AutoTokenizer

    cfg = params()[role]
    return AutoTokenizer.from_pretrained(
        str(source) if source else cfg["name"],
        **({} if source else {"revision": cfg["revision"]}),
        local_files_only=True,
    )


class OnnxEncoder:
    def __init__(self, source, version, provider):
        self.tokenizer = tokenizer_for("encoder", source)
        target = graph_path("encoder", version)
        self.runtime = session(target, provider)
        self.provider = provider

    def encode(self, sentences, normalize_embeddings=True, **kwargs):
        single = isinstance(sentences, str)
        if single:
            sentences = [sentences]
        batch = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        vectors = self.runtime.run(
            None,
            {
                k: batch[k].astype(np.int64)
                for k in ["input_ids", "attention_mask", "token_type_ids"]
            },
        )[0]
        return vectors[0] if single else vectors


class OnnxCrossEncoder:
    def __init__(self, source, version, provider):
        self.tokenizer = tokenizer_for("cross_encoder", source)
        target = graph_path("cross_encoder", version)
        self.runtime = session(target, provider)
        self.provider = provider

    def predict(self, sentences, batch_size=32, **kwargs):
        batch_size = min(batch_size, 4)
        scores = []
        for offset in range(0, len(sentences), batch_size):
            pairs = sentences[offset : offset + batch_size]
            from ledgerlight.pairs import pair_inputs

            batch = pair_inputs(self.tokenizer, pairs, return_tensors="np")
            output = self.runtime.run(
                None,
                {
                    k: batch[k].astype(np.int64)
                    for k in ["input_ids", "attention_mask", "token_type_ids"]
                },
            )[0]
            scores.extend(output.reshape(-1).tolist())
        return np.asarray(scores, dtype=np.float32)
