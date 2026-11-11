import json

import pytest

from semantic_find import cli


def run(capsys, *args):
    code = cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_query_words_and_paths_are_told_apart(notes):
    assert cli._split_search(["renew", "passport", str(notes)]) == ["search", "renew", "passport", "--in", str(notes)]
    assert cli._split_search(["passport", "-k", "3", "--json"]) == ["search", "passport", "-k", "3", "--json"]
    assert cli._split_search(["index", str(notes)]) == ["index", str(notes)]
    assert cli._split_search(["notes"]) == ["search", "notes"], "a first word is the query even if a folder has its name"
    missing = str(notes / "missing")
    assert cli._split_search(["x", missing]) == ["search", "x", "--in", missing], "reported as missing, not searched for"
    assert cli._split_search(["client/server", "design"]) == ["search", "client/server", "design"]


def test_search_by_meaning_from_the_command_line(notes, model_path, capsys):
    code, out, err = run(capsys, "flight", "document", str(notes), "--model", str(model_path))
    assert code == 0 and "travel/passport.md:" in out.replace("\\", "/") and "meaning" in out
    assert "index of" in err and "updated" in err
    code, out, err = run(capsys, "flight", "document", str(notes), "--model", str(model_path))
    assert err == "", "nothing changed: nothing to say"


def test_without_the_model_it_searches_keywords_and_says_why(notes, capsys):
    code, out, err = run(capsys, "embassy", str(notes))
    assert code == 0 and "passport.md" in out and "sfind models download" in err


def test_json_files_and_nothing_found(notes, model_path, capsys):
    code, out, _ = run(capsys, "oven", str(notes), "--model", str(model_path), "--json")
    data = json.loads(out)
    assert code == 0 and data[0]["path"].endswith("bread.md") and data[0]["words"] == ["oven"] and data[0]["line"] == 1
    code, out, _ = run(capsys, "oven", str(notes), "--mode", "keyword", "-l")
    assert out.strip().endswith("bread.md") and len(out.strip().splitlines()) == 1
    code, out, err = run(capsys, "xylophone", str(notes), "--mode", "keyword")
    assert code == 1 and "nothing found" in err


def test_errors_are_plain(capsys, tmp_path):
    code, _, err = run(capsys, "passport", str(tmp_path / "missing"), "--mode", "keyword")
    assert code == 2 and "no such file or folder" in err
    code, _, err = run(capsys, "x", str(tmp_path), "--model", "no-such-model")
    assert code == 2 and "unknown model" in err


def test_status_and_forget(notes, capsys):
    run(capsys, "index", str(notes), "--keyword-only")
    code, out, _ = run(capsys, "status", str(notes))
    assert "5 files" in out and "no model" in out
    code, out, _ = run(capsys, "forget", str(notes))
    assert "index deleted" in out
    code, out, _ = run(capsys, "status", str(notes))
    assert "not indexed yet" in out


def test_models_lists_what_is_there(capsys):
    code, out, _ = run(capsys, "models")
    assert "potion-multilingual-128M (default)" in out and "ollama:" in out


@pytest.mark.parametrize("text,words,expect", [
    ("# Bread\nKnead the dough", ["dough"], "Bread — Knead the dough"),
    ("# comment\ncode()", [], "# comment code()"),
])
def test_snippets_show_the_heading_as_a_heading(text, words, expect):
    heading = "Bread" if "Bread" in text else ""
    assert cli.snippet(text, words, heading=heading) == expect


def test_snippets_are_cut_by_width_on_screen():
    zh = "今天走了七公里。" * 60
    en = "walked seven km. " * 60
    assert len(cli.snippet(zh, [], width=40)) == 21  # 20 wide characters and the ellipsis
    assert len(cli.snippet(en, [], width=40)) == 41
