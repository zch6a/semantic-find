import os
import shutil
import subprocess

import pytest

from semantic_find.embed import StaticModel
from semantic_find.files import discover
from semantic_find.index import Index


def rel(root, paths):
    return sorted(p.relative_to(root).as_posix() for p in paths)


def test_clutter_hidden_folders_and_ignored_files_are_left_out(notes):
    (notes / ".sfindignore").write_text("# comment\n*.py\n", encoding="utf-8")
    found = rel(notes, discover(notes))
    assert found == ["bread.md", "photo.png", "travel/passport.md", "汽车.txt"]


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_in_a_git_repository_gitignore_is_followed(notes):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q", str(notes)], check=True, env=env)
    (notes / ".gitignore").write_text("bread.md\n", encoding="utf-8")
    found = rel(notes, discover(notes))
    assert "bread.md" not in found and ".gitignore" in found and "src/net.py" in found
    assert not any(f.startswith("node_modules/") for f in found), "even when it is not in .gitignore"


def test_an_index_is_built_and_only_changes_are_read_again(notes, model_path):
    model = StaticModel(model_path)
    idx = Index(notes)
    st = idx.update(model, "fake")
    assert (st.files, st.read, st.skipped) == (5, 5, 1)  # the PNG has no text
    assert idx.counts()["embedded"] == idx.counts()["passages"] > 0
    again = idx.update(model, "fake")
    assert again.read == 0 and again.embedded == 0
    (notes / "bread.md").write_text("# Bread\n\nRye flour needs a longer rise.\n", encoding="utf-8")
    (notes / "汽车.txt").unlink()
    third = idx.update(model, "fake")
    assert (third.read, third.removed) == (1, 1)
    assert not idx.search("轮胎", model) and idx.search("rye", model)[0].path == "bread.md"
    idx.close()


def test_a_new_model_embeds_everything_again(notes, model_path, tmp_path):
    from fakemodel import make_model

    idx = Index(notes)
    idx.update(None)
    assert idx.counts()["embedded"] == 0, "keyword-only needs no model"
    n = idx.update(StaticModel(model_path), "fake").embedded
    assert n == idx.counts()["passages"]
    assert idx.update(StaticModel(make_model(tmp_path / "other")), "other").embedded == n
    assert idx.meta("model") == "other"
    idx.close()


def test_keywords_find_words_identifiers_and_chinese(notes):
    idx = Index(notes)
    idx.update(None)
    assert [h.path for h in idx.search("embassy photos", None, mode="keyword")] == ["travel/passport.md"]
    assert idx.search("retry", None, mode="keyword")[0].path == "src/net.py", "fetchWithRetry is split into its words"
    hit = idx.search("汽车轮胎", None, mode="keyword")[0]
    assert hit.path == "汽车.txt" and "轮胎" in hit.words
    assert idx.search("passport", None, mode="keyword")[0].path == "travel/passport.md", "not node_modules, not hidden"
    idx.close()


def test_meaning_finds_what_shares_no_words(notes, model_path):
    model = StaticModel(model_path)
    idx = Index(notes)
    idx.update(model, "fake")
    hits = idx.search("flight document", model, mode="meaning")  # neither word is in any file
    assert hits[0].path == "travel/passport.md" and hits[0].similarity > 0.5
    assert all(h.path != "bread.md" for h in hits), "far below the best match: not listed at all"
    hybrid = idx.search("oven recipe", model)
    assert hybrid[0].path == "bread.md" and hybrid[0].words == ["oven"]
    idx.close()


def test_one_result_per_file_unless_all_are_asked_for(notes, model_path):
    model = StaticModel(model_path)
    idx = Index(notes)
    idx.update(model, "fake")
    best = idx.search("passport visa", model)
    assert [h.path for h in best].count("travel/passport.md") == 1 and best[0].more == 1
    every = idx.search("passport visa", model, per_file=False)
    assert [h.path for h in every].count("travel/passport.md") == 2
    assert [h.path for h in idx.search("passport visa", model, glob="*.py")] == []
    assert sorted(h.line for h in every if h.path == "travel/passport.md") == [1, 5]
    idx.close()
