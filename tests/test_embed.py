import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from fakemodel import make_model, write_safetensors

from semantic_find.embed import EmbedError, OllamaModel, StaticModel, download, is_downloaded, is_local, load, read_safetensors


def test_safetensors_are_read_with_numpy(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_safetensors(tmp_path / "m.safetensors", {"embeddings": a, "half": a.astype(np.float16)})
    t = read_safetensors(tmp_path / "m.safetensors")
    assert np.array_equal(t["embeddings"], a) and t["half"].dtype == np.float16


def test_words_of_a_topic_are_close_and_unknown_words_are_dropped(model_path):
    m = StaticModel(model_path)
    v = m.encode(["passport renewal", "visa for the trip", "bake bread", "zzz qqq", ""])
    assert v[0] @ v[1] > 0.8 and v[0] @ v[2] < 0.3
    assert np.allclose(np.linalg.norm(v[:3], axis=1), 1.0)
    assert not v[3].any() and not v[4].any(), "only unknown words, or nothing: no vector, not a random one"


def test_mapping_and_weights_give_the_same_vectors(tmp_path):
    plain = StaticModel(make_model(tmp_path / "a")).encode(["passport visa", "bread"])
    mapped = StaticModel(make_model(tmp_path / "b", mapping=True, weights=True)).encode(["passport visa", "bread"])
    half = StaticModel(make_model(tmp_path / "c", dtype=np.float16)).encode(["passport visa", "bread"])
    assert np.allclose(plain, mapped, atol=1e-6) and np.allclose(plain, half, atol=1e-2)


def test_loading_names_what_is_missing(tmp_path, model_path):
    with pytest.raises(EmbedError, match="not downloaded yet"):
        load("potion-base-8M")
    with pytest.raises(EmbedError, match="unknown model"):
        load("no-such-model")
    assert load(str(model_path)).dim == 8
    (model_path / "model.safetensors").write_bytes(b"\x10\x00\x00\x00\x00\x00\x00\x00{not json}     ")
    with pytest.raises(EmbedError, match="could not be loaded"):
        StaticModel(model_path)


def test_only_ollama_on_this_machine_is_used():
    assert is_local("http://localhost:11434") and is_local("http://127.0.0.1:1") and is_local("http://[::1]:1")
    assert not is_local("http://127.0.0.1.example.com") and not is_local("http://10.0.0.2:11434")
    with pytest.raises(EmbedError, match="not on this machine"):
        OllamaModel("bge-m3", url="http://10.0.0.2:11434")
    with pytest.raises(EmbedError, match="cloud model"):
        OllamaModel("gpt-oss:20b-cloud")


class Server:
    """A stand-in for Hugging Face (listing and downloads, with ranges) or for Ollama's /api/embed."""

    def __init__(self, files=None, listing=None, embed=None):
        self.files, self.listing, self.embed, self.requests = files or {}, listing, embed, []
        me = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                me.requests.append((self.path, self.headers.get("Range")))
                if self.path.startswith("/api/models/"):
                    return self._send(200, json.dumps(me.listing).encode())
                name = self.path.rsplit("/", 1)[-1]
                data = me.files.get(name)
                if data is None:
                    return self._send(404, b"no")
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    return self._send(206, data[start:])
                return self._send(200, data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                me.requests.append((self.path, body))
                return self._send(200, json.dumps({"embeddings": [me.embed(t) for t in body["input"]]}).encode())

            def _send(self, code, data):
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def pin(monkeypatch, files, lie_about=None):
    """Replace the built-in potion-base-8M by the fake model's files, pinned by size and SHA-256."""
    import dataclasses

    from semantic_find import embed

    pinned = {f: (len(d), "0" * 64 if f == lie_about else hashlib.sha256(d).hexdigest()) for f, d in files.items()}
    fake = dataclasses.replace(embed.MODELS["potion-base-8M"], files=pinned, revision="abc123")
    monkeypatch.setitem(embed.MODELS, "potion-base-8M", fake)


def test_a_model_is_downloaded_checked_and_resumed(tmp_path, monkeypatch, home):
    src = make_model(tmp_path / "src")
    files = {f: (src / f).read_bytes() for f in ("config.json", "tokenizer.json", "model.safetensors")}
    hf = Server(files)
    pin(monkeypatch, files)
    monkeypatch.setenv("HF_ENDPOINT", hf.url)
    try:
        part = home / "models" / "potion-base-8M" / "model.safetensors.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(files["model.safetensors"][:100])  # an interrupted download
        download("potion-base-8M")
        assert is_downloaded("potion-base-8M") and not part.exists()
        assert ("/minishlab/potion-base-8M/resolve/abc123/model.safetensors", "bytes=100-") in hf.requests, "at the pinned revision"
        assert not any(path.startswith("/api/") for path, _ in hf.requests), "no API call: the sizes and hashes are pinned"
        assert load("potion-base-8M").encode(["passport"]).shape == (1, 8)
    finally:
        hf.close()


def test_a_corrupt_download_is_refused(tmp_path, monkeypatch, home):
    src = make_model(tmp_path / "src")
    files = {f: (src / f).read_bytes() for f in ("config.json", "tokenizer.json", "model.safetensors")}
    hf = Server(files)
    pin(monkeypatch, files, lie_about="model.safetensors")
    monkeypatch.setenv("HF_ENDPOINT", hf.url)
    try:
        with pytest.raises(EmbedError, match="SHA-256"):
            download("potion-base-8M")
        assert not is_downloaded("potion-base-8M")
    finally:
        hf.close()


def test_an_unreachable_hub_suggests_a_mirror(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    with pytest.raises(EmbedError, match="HF_ENDPOINT"):
        download("potion-base-8M")


def test_ollama_embeddings_are_normalized_and_never_proxied(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")  # a proxy that does not exist: using it would fail
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    ollama = Server(embed=lambda t: [3.0, 4.0] if "passport" in t else [0.0, 2.0])
    try:
        m = OllamaModel("bge-m3", url=ollama.url)
        v = m.encode(["passport", "bread"])
        assert m.dim == 2 and np.allclose(v, [[0.6, 0.8], [0.0, 1.0]])
        assert ollama.requests[-1][1]["model"] == "bge-m3"
    finally:
        ollama.close()


def test_the_tokenizer_facts_are_worked_out_once(model_path):
    import os

    first = StaticModel(model_path)
    cache = model_path / "sfind-facts.json"
    assert cache.is_file() and json.loads(cache.read_text())["unk"] == first.unk == 0
    cache.write_text(json.dumps({**json.loads(cache.read_text()), "median_token_length": 99}))
    assert StaticModel(model_path).median_token_length == 99, "read from the file, not worked out again"
    st = (model_path / "tokenizer.json").stat()
    os.utime(model_path / "tokenizer.json", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert StaticModel(model_path).median_token_length != 99, "a changed tokenizer is worked out again"


def test_unknown_words_do_not_pull_the_vector_away(model_path):
    m = StaticModel(model_path)
    with_unknown, plain = m.encode(["passport zzz qqq www", "passport"])
    assert np.allclose(with_unknown, plain, atol=1e-6)
