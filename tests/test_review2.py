"""Regression tests for the second independent review (POLISH_LOG, round 3). Each one failed before its fix."""

import numpy as np
import pytest

from semantic_find import cli
from semantic_find import index as ix
from semantic_find.embed import StaticModel


def run(capsys, *args):
    code = cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def folder(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir(parents=True)
    for rel, text in files.items():
        (d / rel).write_text(text, encoding="utf-8")
    return d


def test_the_file_with_the_exact_word_is_never_hidden(tmp_path, model_path, capsys):
    d = folder(tmp_path, "d", {"net.md": "retry request timeout network pause", "garage.md": "backoff car engine brakes winter tyres"})
    code, out, _ = run(capsys, "backoff", str(d), "--model", str(model_path), "-q")
    assert "garage.md" in out, "grep would find it"


def test_several_folders_are_scored_on_one_scale(tmp_path, model_path, capsys):
    a = folder(tmp_path, "p/a", {"passport.md": "passport renewal embassy", "flight.md": "flight trip visa travel"})
    b = folder(tmp_path, "p/b", {"bakery.md": "bread oven dough flour trip"})
    _, together, _ = run(capsys, "passport renewal", str(tmp_path / "p"), "--model", str(model_path), "-q")
    _, apart, _ = run(capsys, "passport renewal", str(a), str(b), "--model", str(model_path), "-q")

    def order(out):
        return [line.split(":")[0].replace("\\", "/").rsplit("/", 1)[-1] for line in out.splitlines() if line and not line.startswith(" ")]

    assert order(apart) == order(together)


def test_a_word_absent_from_the_folder_counts_against_a_match(tmp_path, model_path, monkeypatch, capsys):
    rng = np.random.default_rng(3)
    words = ["bread", "oven", "dough", "retry", "timeout", "request", "car", "tyres", "engine", "the", "a", "for", "it", "is"]
    files = {f"n{i}.md": " ".join(rng.choice(words, 10)) for i in range(40)}
    files["piano.md"] = "Piano lessons on Tuesday"
    d = folder(tmp_path, "d", files)
    monkeypatch.setattr(ix, "STANDOUT_FROM", 10)
    code, out, err = run(capsys, "xylophone lessons", str(d), "--model", str(model_path), "-q")
    assert code == 1, "one word of two in common, and xylophone is nowhere: not a match"
    idx = ix.Index(d)
    pid = idx.keyword("lessons")[0]
    assert idx.coverage("xylophone lessons", [pid])[pid] < 0.5
    idx.close()


def test_accents_are_folded_and_weigh_like_any_word(tmp_path, capsys):
    d = folder(tmp_path, "d", {"fr.md": "le café du coin", "menu.md": "the menu of the day"})
    code, out, _ = run(capsys, "cafe", str(d), "--mode", "keyword", "-q")
    assert code == 0 and "fr.md" in out, "cafe finds café"
    idx = ix.Index(d)
    kw = idx.keyword("café menu")
    cover = idx.coverage("café menu", kw)
    assert len(cover) == 2 and all(0.3 < c < 0.7 for c in cover.values()), "each has half of the query, not all of it"
    idx.close()


def test_g_only_filters_and_never_adds_results(tmp_path, model_path, monkeypatch):
    rng = np.random.default_rng(3)
    words = ["bread", "oven", "dough", "retry", "timeout", "request", "car", "tyres", "engine", "the", "a", "for", "it", "is"]
    files = {f"n{i}.{'md' if i < 5 else 'txt'}": " ".join(rng.choice(words, 10)) for i in range(45)}
    d = folder(tmp_path, "d", files)
    monkeypatch.setattr(ix, "STANDOUT_FROM", 10)
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    assert all(h.weak for h in idx.search("visa trip", model, glob="*.md")), "the background is the whole folder"
    idx.close()


def test_the_same_folder_twice_or_nested_lists_each_result_once(tmp_path, model_path, capsys):
    a = folder(tmp_path, "p/a", {"passport.md": "passport renewal embassy", "flight.md": "flight trip visa travel"})
    for args in ([str(a), str(a)], [str(tmp_path / "p"), str(a)]):
        _, out, _ = run(capsys, "passport renewal", *args, "--model", str(model_path), "-q")
        assert out.count("passport.md") == 1, args


def test_passages_without_vectors_are_found_by_their_words(tmp_path, model_path, capsys):
    d = folder(tmp_path, "d", {"a.md": "passport renewal", "b.md": "bread oven"})
    run(capsys, "index", str(d), "--model", str(model_path))
    (d / "new.md").write_text("The embassy phone number is on the website", encoding="utf-8")
    run(capsys, "index", str(d), "--keyword-only")
    code, out, _ = run(capsys, "embassy phone number", str(d), "--model", str(model_path), "--no-update", "-q")
    assert "new.md" in out.splitlines()[0]


def test_a_file_name_that_is_not_unicode_is_skipped_not_fatal(tmp_path, capsys):
    d = folder(tmp_path, "d", {"good.md": "passport"})
    try:
        (d / "bad\udc80.md").write_text("passport", encoding="utf-8")
    except (OSError, UnicodeEncodeError):
        pytest.skip("this file system cannot make such a name")
    code, out, _ = run(capsys, "passport", str(d), "--mode", "keyword", "-q")
    assert code == 0 and "good.md" in out


def test_a_copy_file_keeps_its_own_other_passages(tmp_path, model_path):
    long_x = "passport renewal embassy photos " * 30  # long enough to be a passage of its own
    d = folder(tmp_path, "d", {"a.md": long_x, "b.md": long_x + "\n\n" + "visa trip flight travel " * 30})
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    hits = idx.search("passport visa trip", model, per_file=True)
    first = next(h for h in hits if "passport" in h.text and "visa" not in h.text)
    assert first.more == 0, "b.md's other passage is not counted as a.md's"
    assert any(h.path == "b.md" and "visa" in h.text for h in hits), "b.md's own passage is shown"
    idx.close()


def test_bad_k_and_meaning_without_vectors_are_explained(tmp_path, capsys):
    d = folder(tmp_path, "d", {"a.md": "passport"})
    code, _, err = run(capsys, "passport", str(d), "-k", "0")
    assert code == 2 and "-k must be at least 1" in err
    run(capsys, "index", str(d), "--keyword-only")
    from fakemodel import make_model

    m = make_model(tmp_path / "m")
    code, _, err = run(capsys, "passport", str(d), "--mode", "meaning", "--model", str(m), "--no-update")
    assert code == 2 and "no vectors yet" in err


def test_weak_matches_go_when_there_is_a_strong_one(tmp_path, model_path, monkeypatch):
    d = folder(tmp_path, "d", {"a.md": "passport renewal", "b.md": "visa trip flight travel"})  # b: close in meaning, no words
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    monkeypatch.setattr(ix, "STANDOUT_FROM", 1)
    monkeypatch.setattr(ix, "MIN_STANDOUT", 100.0)  # nothing found by meaning alone stands out
    assert [h.path for h in idx.search("passport renewal", model)] == ["a.md"], "a.md has the words: b.md is weak, and goes"
    idx.close()


def test_a_meaning_match_far_below_the_best_is_not_shown(tmp_path, model_path, monkeypatch):
    d = folder(tmp_path, "d", {"a.md": "visa trip flight travel", "b.md": "visa bread oven dough flour knead"})
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    monkeypatch.setattr(ix, "SCORE_FLOOR", 0.0)  # the relative floor on similarity alone
    hits = idx.search("flight trip", model, mode="meaning")
    assert [h.path for h in hits] == ["a.md"], [(h.path, round(h.similarity, 2)) for h in hits]
    idx.close()


def test_a_passage_repeated_in_one_file_is_not_its_own_copy(tmp_path, model_path):
    long_x = "passport renewal embassy photos " * 30
    d = folder(tmp_path, "d", {"a.md": long_x + "\n\n" + long_x, "b.md": long_x})
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    hits = idx.search("passport renewal", model, per_file=False)
    first = hits[0]
    assert first.path not in first.same and first.same.count("b.md") <= 1
    idx.close()


def test_a_file_whose_reader_fails_in_any_way_is_skipped(tmp_path, monkeypatch, capsys):
    d = folder(tmp_path, "d", {"good.md": "passport", "odd.md": "passport too"})
    real = ix.read_text

    def flaky(path):
        if path.name == "odd.md":
            raise ValueError("a reader's own error, not an OSError")
        return real(path)

    monkeypatch.setattr(ix, "read_text", flaky)
    code, out, _ = run(capsys, "passport", str(d), "--mode", "keyword", "-q")
    assert code == 0 and "good.md" in out


def test_g_filters_before_the_meaning_candidates_are_cut(tmp_path, model_path, monkeypatch):
    files = {f"n{i}.txt": "visa trip flight travel" for i in range(8)}
    files["plan.md"] = "visa trip"
    d = folder(tmp_path, "d", files)
    model = StaticModel(model_path)
    idx = ix.Index(d)
    idx.update(model, "fake")
    monkeypatch.setattr(ix, "CANDIDATES", 3)
    assert [h.path for h in idx.search("flight travel", model, mode="meaning", glob="*.md")] == ["plan.md"]
    idx.close()


def test_chinese_coverage_is_counted_by_characters(tmp_path):
    d = folder(tmp_path, "d", {"db.md": "数据库配置要写在启动脚本里"})
    idx = ix.Index(d)
    idx.update(None)
    pid = idx.keyword("怎么配置数据库")[0]
    assert idx.coverage("怎么配置数据库", [pid])[pid] == pytest.approx(1.0), "every character but the filler is there"
    idx.close()
