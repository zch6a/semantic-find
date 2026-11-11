"""What a search shows: copies said once, weak matches left out, nothing for what is not there."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from semantic_find import cli
from semantic_find import index as ix
from semantic_find.embed import StaticModel


def run(capsys, *args):
    code = cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_copies_of_a_passage_are_shown_once(tmp_path, model_path, capsys):
    d = tmp_path / "notes"
    d.mkdir()
    for i in range(4):
        (d / f"copy-{i}.md").write_text("The passport renewal takes six weeks.", encoding="utf-8")
    (d / "other.md").write_text("Renew the visa before the trip.", encoding="utf-8")
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    hits = idx.search("passport renewal", model)
    first = next(h for h in hits if h.path.startswith("copy-"))
    assert sum(h.path.startswith("copy-") for h in hits) == 1 and len(first.same) == 3
    idx.close()
    code, out, _ = run(capsys, "passport renewal", str(d), "--model", str(model_path), "-q")
    assert "the same text in 3 other files" in out
    code, out, _ = run(capsys, "passport renewal", str(d), "--model", str(model_path), "-q", "-l")
    assert sum("copy-" in line for line in out.splitlines()) == 4, "-l lists every file that matches, copies included"
    code, out, _ = run(capsys, "passport renewal", str(d), "--model", str(model_path), "-q", "--json")
    assert any(len(r["same"]) == 3 for r in json.loads(out))


def test_weak_matches_after_a_strong_one_are_left_out(tmp_path, model_path, monkeypatch):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "strong.md").write_text("passport embassy photos renewal", encoding="utf-8")  # the words and the meaning
    (d / "loose.md").write_text("visa bread oven dough", encoding="utf-8")  # half about travel, no word in common
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    monkeypatch.setattr(ix, "RELATIVE_FLOOR", 0.0)  # this test is about the score floor alone
    monkeypatch.setattr(ix, "SCORE_FLOOR", 0.0)
    assert [h.path for h in idx.search("passport embassy photos", model)] == ["strong.md", "loose.md"]
    monkeypatch.setattr(ix, "SCORE_FLOOR", 0.6)
    assert [h.path for h in idx.search("passport embassy photos", model)] == ["strong.md"]
    idx.close()


def test_in_a_large_folder_a_match_by_meaning_must_stand_out(tmp_path, model_path, monkeypatch):
    """Among many passages that have nothing to do with the query, the best of them is only noise: with
    enough passages for a background, it is not shown. A passage really about it stands out."""
    model = StaticModel(model_path)
    d = tmp_path / "big"
    d.mkdir()
    rng = np.random.default_rng(3)
    words = ["bread", "oven", "dough", "retry", "timeout", "request", "car", "tyres", "engine", "the", "a", "for", "it", "is"]
    for i in range(40):
        (d / f"n{i}.md").write_text(" ".join(rng.choice(words, 10)), encoding="utf-8")
    idx = ix.Index(d)
    idx.update(model, "fake")
    monkeypatch.setattr(ix, "MIN_SIMILARITY", -1.0)
    monkeypatch.setattr(ix, "RELATIVE_FLOOR", 0.0)
    monkeypatch.setattr(ix, "STANDOUT_FROM", 10**9)
    assert idx.search("visa trip", model, mode="meaning"), "without the test, the noise is listed"
    monkeypatch.setattr(ix, "STANDOUT_FROM", 10)
    assert idx.search("visa trip", model, mode="meaning") == [], "with it, nothing: none stands out"
    (d / "trip.md").write_text("flight visa passport embassy travel", encoding="utf-8")
    idx.update(model, "fake")
    assert [h.path for h in idx.search("visa trip", model, mode="meaning")] == ["trip.md"]
    idx.close()


def test_one_passage_is_said_in_the_singular(tmp_path, model_path, capsys):
    d = tmp_path / "one"
    d.mkdir()
    (d / "a.md").write_text("passport", encoding="utf-8")
    _, _, err = run(capsys, "passport", str(d), "--model", str(model_path))
    assert "1 passage embedded" in err and "1 passages" not in err


def test_the_corpus_maker_makes_a_folder_that_indexes(tmp_path, capsys):
    maker = Path(__file__).resolve().parent.parent / "eval" / "make_corpus.py"
    out = subprocess.run([sys.executable, str(maker), str(tmp_path / "c"), "--files", "60"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    code, found, _ = run(capsys, "insurer plumber invoice", str(tmp_path / "c"), "--mode", "keyword", "-q", "-l")
    assert code == 0 and found.splitlines()[0].replace("\\", "/").endswith("archive/2023/house/insurance-claim.md")
