import numpy as np


def pair_inputs(tokenizer, pairs, return_tensors="pt"):
    return tokenizer(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        truncation="only_second",
        max_length=512,
        padding=True,
        return_tensors=return_tensors,
    )


def torch_pair_scores(model, pairs, batch_size=32):
    import torch

    model.model.eval()
    values = []
    device = next(model.model.parameters()).device
    with torch.inference_mode():
        for offset in range(0, len(pairs), batch_size):
            batch = pair_inputs(model.tokenizer, pairs[offset : offset + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            values.extend(model.model(**batch).logits.reshape(-1).float().cpu().tolist())
    return np.asarray(values, dtype=np.float32)
