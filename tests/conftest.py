import pytest
from fakemodel import make_model


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test gets its own data folder: no test touches the real models or indexes."""
    h = tmp_path / "home"
    monkeypatch.setenv("SEMANTIC_FIND_HOME", str(h))
    monkeypatch.delenv("SFIND_MODEL", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    return h


@pytest.fixture
def model_path(tmp_path):
    return make_model(tmp_path / "fake-model")


@pytest.fixture
def notes(tmp_path):
    """A small folder of synthetic notes, in English and Chinese, with code and clutter."""
    d = tmp_path / "notes"
    (d / "travel").mkdir(parents=True)
    (d / "src").mkdir()
    (d / "node_modules" / "lib").mkdir(parents=True)
    (d / ".hidden").mkdir()
    (d / "travel" / "passport.md").write_text(
        "# Passport\n\nRenewal takes about six weeks. Bring the old passport and two photos to the embassy.\n\n"
        "# Visa\n\nA visa for the trip is applied for online.\n", encoding="utf-8")
    (d / "bread.md").write_text("# Bread\n\nKnead the dough for ten minutes and bake in a hot oven.\n", encoding="utf-8")
    (d / "汽车.txt").write_text("冬天要换轮胎，每个月检查一次汽车的轮胎。\n", encoding="utf-8")
    (d / "src" / "net.py").write_text("def fetchWithRetry(url):\n    # retry with exponential backoff when a request fails\n    pass\n", encoding="utf-8")
    (d / "node_modules" / "lib" / "index.js").write_text("passport passport passport", encoding="utf-8")
    (d / ".hidden" / "secret.md").write_text("passport", encoding="utf-8")
    (d / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00passport")
    return d
