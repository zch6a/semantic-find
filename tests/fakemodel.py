"""A tiny Model2Vec-format model for tests: a word-level tokenizer and hand-made vectors in which words
of one topic point the same way, so that "search by meaning" can be tested without a download."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

TOPICS = {
    "travel": ["passport", "visa", "travel", "document", "trip", "flight", "renewal", "renew", "embassy", "护照", "签证", "旅行"],
    "cooking": ["bread", "dough", "bake", "oven", "knead", "flour", "recipe", "temperature", "面包", "烤箱", "食谱"],
    "network": ["retry", "backoff", "request", "timeout", "fails", "exponential", "pause", "network", "重试", "超时", "请求"],
    "car": ["car", "tyres", "tires", "engine", "winter", "brakes", "汽车", "轮胎"],
}


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, offset, blobs = {}, 0, []
    names = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16", np.dtype(np.int64): "I64", np.dtype(np.int32): "I32"}
    for key, arr in tensors.items():
        data = np.ascontiguousarray(arr).tobytes()
        header[key] = {"dtype": names[arr.dtype], "shape": list(arr.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def make_model(folder: Path, *, weights: bool = False, mapping: bool = False, dtype=np.float32) -> Path:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

    folder.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0}
    vectors = [np.full(len(TOPICS) + 4, 0.3)]  # [UNK]: not zero, as in real models, so that keeping it would show
    rng = np.random.default_rng(7)
    for t, (_, words) in enumerate(TOPICS.items()):
        for w in words:
            v = np.zeros(len(TOPICS) + 4)
            v[t] = 1.0
            v[len(TOPICS):] = rng.normal(0, 0.15, 4)  # a little noise: words of a topic are close, not identical
            vocab[w] = len(vocab)
            vectors.append(v)
    for filler in ["the", "a", "how", "do", "i", "my", "what", "for", "when", "is", "it"]:
        vocab[filler] = len(vocab)
        vectors.append(rng.normal(0, 0.05, len(TOPICS) + 4))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Split(r"[一-鿿]{2}", "isolated")])
    tok.save(str(folder / "tokenizer.json"))
    emb = np.asarray(vectors, dtype=dtype)
    tensors = {"embeddings": emb}
    if mapping:  # rows stored in reverse order, with a mapping from token id to row
        tensors["embeddings"] = emb[::-1].copy()
        tensors["mapping"] = np.arange(len(vocab) - 1, -1, -1, dtype=np.int64)
    if weights:
        tensors["weights"] = np.ones(len(vocab), dtype=np.float32)
    write_safetensors(folder / "model.safetensors", tensors)
    (folder / "config.json").write_text(json.dumps({"normalize": True, "max_length": 64}), encoding="utf-8")
    return folder
