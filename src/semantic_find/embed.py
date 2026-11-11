"""Turning passages into vectors, on this machine.

The default is a *static* embedding model (Model2Vec, distilled from a sentence transformer): each token
has a vector, and a passage is the average of its tokens' vectors. No neural network runs, so it needs
no GPU and no PyTorch, and embeds thousands of passages a second on a laptop CPU. Its quality is below a
full transformer's, which is why search also matches keywords and fuses the two rankings.

A model is downloaded once from Hugging Face (or the mirror in `HF_ENDPOINT`), only when asked, and
checked against the size and SHA-256 that Hugging Face lists. After that nothing goes over the network.

With Ollama running, any of its embedding models can be used instead (`ollama:bge-m3`); only a server on
this machine is accepted, since the passages are the contents of your files.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelInfo:
    name: str
    repo: str
    languages: str
    size_mb: int
    note: str


MODELS = {m.name: m for m in [
    ModelInfo("potion-multilingual-128M", "minishlab/potion-multilingual-128M", "101 languages, Chinese and English among them", 507,
              "the default: one model for documents in any language"),
    ModelInfo("potion-retrieval-32M", "minishlab/potion-retrieval-32M", "English", 125, "trained for search; smaller and better for English-only files"),
    ModelInfo("potion-base-8M", "minishlab/potion-base-8M", "English", 30, "the smallest; for a quick try"),
]}
DEFAULT_MODEL = "potion-multilingual-128M"
FILES = ("config.json", "tokenizer.json", "model.safetensors")


class EmbedError(RuntimeError):
    pass


def data_home() -> Path:
    """Where models and indexes live: SEMANTIC_FIND_HOME, else the platform's per-user data folder."""
    if os.environ.get("SEMANTIC_FIND_HOME"):
        return Path(os.environ["SEMANTIC_FIND_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "semantic-find"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "semantic-find"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "semantic-find"


def model_dir(name: str) -> Path:
    return data_home() / "models" / name


def is_downloaded(name: str) -> bool:
    d = model_dir(name)
    return all((d / f).is_file() for f in FILES)


# ---------------------------------------------------------------- safetensors, read with numpy

DTYPES = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "I32": np.int32, "I64": np.int64, "BF16": None}


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """The tensors of a .safetensors file, memory-mapped: an 8-byte header length, a JSON header, then the data."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype = DTYPES.get(meta["dtype"])
        if dtype is None:
            raise EmbedError(f"{path.name}: tensor {key} has an unsupported type {meta['dtype']}")
        start, end = meta["data_offsets"]
        count = int(np.prod(meta["shape"])) if meta["shape"] else 1
        if end - start != count * np.dtype(dtype).itemsize:
            raise EmbedError(f"{path.name}: tensor {key} has the wrong size")
        out[key] = np.memmap(path, dtype=dtype, mode="r", offset=base + start, shape=tuple(meta["shape"]))
    return out


# ---------------------------------------------------------------- the static model

class StaticModel:
    """Model2Vec's inference, reimplemented with numpy and Hugging Face's `tokenizers`: token ids (unknown
    tokens dropped), each mapped and weighted if the model says so, averaged, and normalized."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.path = Path(path)
        try:
            tensors = read_safetensors(self.path / "model.safetensors")
            self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
            config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise EmbedError(f"the model in {self.path} could not be loaded ({e}); download it again") from e
        emb = tensors.get("embeddings", tensors.get("embedding.weight"))
        if emb is None:
            raise EmbedError(f"{self.path}: no embeddings in model.safetensors")
        self.embeddings = emb
        self.weights = tensors.get("weights")
        self.mapping = tensors.get("mapping")
        self.normalize = bool(config.get("normalize", True))
        self.max_length = int(config.get("max_length") or 512)
        self.dim = int(emb.shape[1])
        self.unk = self._unk_id()
        vocab = self.tokenizer.get_vocab()
        self.median_token_length = max(1, int(np.median([len(t) for t in vocab]))) if vocab else 1
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(self.max_length)

    def _unk_id(self) -> int | None:
        model = json.loads(self.tokenizer.to_str())["model"]
        if model.get("type") == "Unigram" and model.get("unk_id") is not None:
            return int(model["unk_id"])
        token = model.get("unk_token")
        return self.tokenizer.token_to_id(token) if token else None

    def encode(self, texts: list[str], batch: int = 1024) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        cut = self.max_length * self.median_token_length  # characters: no point tokenizing what is cut anyway
        for start in range(0, len(texts), batch):
            chunk = [t[:cut] for t in texts[start:start + batch]]
            for i, enc in enumerate(self.tokenizer.encode_batch(chunk, add_special_tokens=False)):
                ids = [t for t in enc.ids if t != self.unk]
                if not ids:
                    continue
                idx = np.asarray(ids)
                rows = self.mapping[idx] if self.mapping is not None else idx
                vecs = np.asarray(self.embeddings[rows], dtype=np.float32)
                if self.weights is not None:
                    vecs = vecs * np.asarray(self.weights[idx], dtype=np.float32)[:, None]
                out[start + i] = vecs.mean(axis=0)
        if self.normalize:
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-32
        return out


# ---------------------------------------------------------------- Ollama, on this machine only

def is_local(url: str) -> bool:
    import ipaddress

    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OllamaModel:
    """An embedding model served by Ollama on this machine (`ollama pull bge-m3`)."""

    def __init__(self, model: str, url: str | None = None, timeout: float = 120):
        self.model = model
        self.url = (url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if "://" not in self.url:
            self.url = "http://" + self.url
        if not is_local(self.url):
            raise EmbedError(f"{self.url} is not on this machine; semantic-find sends the contents of your files only to a local server")
        if self.model.endswith(("-cloud", ":cloud")):
            raise EmbedError(f"\"{self.model}\" is an Ollama cloud model: it runs on Ollama's servers, not here")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy
        self.dim = len(self.encode(["dimension probe"])[0])

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), batch):
            body = json.dumps({"model": self.model, "input": texts[start:start + batch], "truncate": True}).encode()
            req = urllib.request.Request(self.url + "/api/embed", body, {"Content-Type": "application/json"})
            try:
                with self.opener.open(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                hint = f"; run `ollama pull {self.model}`" if e.code == 404 else ""
                raise EmbedError(f"Ollama answered {e.code}: {detail}{hint}") from e
            except (urllib.error.URLError, OSError) as e:
                raise EmbedError(f"Ollama is not answering at {self.url} ({e}); start it, or use a built-in model") from e
            rows.extend(data.get("embeddings") or [])
        if len(rows) != len(texts):
            raise EmbedError("Ollama returned the wrong number of embeddings")
        out = np.asarray(rows, dtype=np.float32)
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-32
        return out


def load(name: str):
    """A model by name: a built-in static model (downloaded already), `ollama:<model>`, or a folder
    holding a Model2Vec model (config.json, tokenizer.json, model.safetensors)."""
    if name.startswith("ollama:"):
        return OllamaModel(name.split(":", 1)[1])
    if Path(name).is_dir():
        return StaticModel(Path(name))
    if name not in MODELS:
        raise EmbedError(f"unknown model \"{name}\"; built in: {', '.join(MODELS)}; or ollama:<name>, or a folder")
    if not is_downloaded(name):
        raise EmbedError(f"the model {name} is not downloaded yet: run `sfind models download {name}` ({MODELS[name].size_mb} MB, once)")
    return StaticModel(model_dir(name))


# ---------------------------------------------------------------- downloading

def endpoint() -> str:
    """Hugging Face, or the mirror in HF_ENDPOINT (for example https://hf-mirror.com)."""
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def _listing(repo: str) -> dict[str, dict]:
    url = f"{endpoint()}/api/models/{repo}/tree/main"
    with urllib.request.urlopen(url, timeout=30) as r:
        return {f["path"]: f for f in json.loads(r.read().decode("utf-8")) if f.get("type") == "file"}


def download(name: str, progress=None) -> Path:
    """Fetch a built-in model's three files, resuming a partial download, and check each one's size and
    (for the large files Hugging Face stores with Git LFS) its SHA-256. Returns the model's folder."""
    if name not in MODELS:
        raise EmbedError(f"unknown model \"{name}\"; built in: {', '.join(MODELS)}")
    info = MODELS[name]
    target = model_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    try:
        listing = _listing(info.repo)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise EmbedError(f"could not reach {endpoint()} ({e}); set HF_ENDPOINT to a mirror, or download the files by hand into {target}") from e
    for file in FILES:
        meta = listing.get(file)
        if meta is None:
            raise EmbedError(f"{info.repo} has no {file}")
        size = int(meta.get("size") or 0)
        sha = (meta.get("lfs") or {}).get("oid")
        final = target / file
        if final.is_file() and final.stat().st_size == size:
            continue
        part = final.with_suffix(final.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(f"{endpoint()}/{info.repo}/resolve/main/{file}", headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as out:
                done = have if r.status == 206 else 0
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if progress:
                        progress(file, done, size)
        except (urllib.error.URLError, OSError) as e:
            raise EmbedError(f"downloading {file} failed ({e}); run the command again to resume") from e
        if part.stat().st_size != size:
            raise EmbedError(f"{file} is {part.stat().st_size} bytes, not {size}; run the command again to resume")
        if sha:
            h = hashlib.sha256()
            with open(part, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != sha:
                part.unlink()
                raise EmbedError(f"{file} does not match its SHA-256 on Hugging Face; it was deleted, run the command again")
        os.replace(part, final)
    return target
