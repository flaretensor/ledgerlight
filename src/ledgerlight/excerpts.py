from functools import lru_cache

from ledgerlight.common import params, path


@lru_cache(maxsize=1)
def window_tokenizer():
    from transformers import AutoTokenizer

    cfg = params()["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["name"],
        revision=cfg["revision"],
        cache_dir=path(".cache/huggingface/hub"),
        local_files_only=True,
        token=False,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Original excerpts require a fast encoder tokenizer")
    return tokenizer


def original_window_text(text, window, tokenizer=None):
    tokenizer = window_tokenizer() if tokenizer is None else tokenizer
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    start, end = window["start"], window["end"]
    if not 0 <= start < end <= len(encoded["input_ids"]):
        raise ValueError("Invalid original passage token span")
    if encoded["input_ids"][start:end] != window["input_ids"]:
        raise ValueError("Original passage token IDs do not match stored window input_ids")
    char_start = encoded["offset_mapping"][start][0]
    char_end = encoded["offset_mapping"][end - 1][1]
    if not 0 <= char_start < char_end <= len(text):
        raise ValueError("Invalid original passage character span")
    return text[char_start:char_end]
