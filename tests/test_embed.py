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


def _listing(files, lie_about=None):
    out = []
    for name, data in files.items():
        entry = {"type": "file", "path": name, "size": len(data)}
        if name.endswith(".safetensors"):
            entry["lfs"] = {"oid": "0" * 64 if name == lie_about else hashlib.sha256(data).hexdigest()}
        out.append(entry)
    return out


def test_a_model_is_downloaded_checked_and_resumed(tmp_path, monkeypatch, home):
    src = make_model(tmp_path / "src")
    files = {f: (src / f).read_bytes() for f in ("config.json", "tokenizer.json", "model.safetensors")}
    hf = Server(files, _listing(files))
    monkeypatch.setenv("HF_ENDPOINT", hf.url)
    try:
        part = home / "models" / "potion-base-8M" / "model.safetensors.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(files["model.safetensors"][:100])  # an interrupted download
        download("potion-base-8M")
        assert is_downloaded("potion-base-8M") and not part.exists()
        assert ("/minishlab/potion-base-8M/resolve/main/model.safetensors", "bytes=100-") in hf.requests
        assert load("potion-base-8M").encode(["passport"]).shape == (1, 8)
    finally:
        hf.close()


def test_a_corrupt_download_is_refused(tmp_path, monkeypatch, home):
    src = make_model(tmp_path / "src")
    files = {f: (src / f).read_bytes() for f in ("config.json", "tokenizer.json", "model.safetensors")}
    hf = Server(files, _listing(files, lie_about="model.safetensors"))
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
