"""Regression tests for what the first independent review found (POLISH_LOG, round 1). Each one failed
before its fix."""

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from fakemodel import make_model, write_safetensors
from pdfmaker import make_pdf

from semantic_find import cli
from semantic_find import index as ix
from semantic_find.embed import EmbedError, StaticModel, is_local, read_safetensors
from semantic_find.files import discover
from semantic_find.text import decode, read_text


def run(capsys, *args):
    code = cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_damaged_files_are_skipped_and_the_rest_is_indexed(tmp_path, capsys):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "good.md").write_text("passport renewal", encoding="utf-8")
    (d / "empty.pdf").write_bytes(b"")
    (d / "truncated.pdf").write_bytes(make_pdf(["half"])[:120])
    (d / "odd.ipynb").write_text('{"cells": null}', encoding="utf-8")
    (d / "cells.ipynb").write_text('{"cells": [null, {"source": 5}]}', encoding="utf-8")
    code, out, err = run(capsys, "passport", str(d), "--mode", "keyword")
    assert code == 0 and "good.md" in out and "Traceback" not in err


def test_an_interrupted_index_keeps_what_it_has_done(tmp_path, monkeypatch):
    d = tmp_path / "docs"
    d.mkdir()
    for i in range(5):
        (d / f"n{i}.md").write_text(f"note {i}", encoding="utf-8")
    monkeypatch.setattr(ix, "COMMIT_EVERY", 2)
    real = ix.read_text
    calls = []

    def flaky(path):
        calls.append(path)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return real(path)

    monkeypatch.setattr(ix, "read_text", flaky)
    idx = ix.Index(d)
    with pytest.raises(KeyboardInterrupt):
        idx.update(None)
    idx.close()
    assert ix.Index(d).counts()["files"] >= 2, "the files before the last commit are kept"


def test_a_second_sfind_searches_while_the_first_updates(notes, capsys):
    run(capsys, "index", str(notes), "--keyword-only")
    (notes / "new.md").write_text("passport photos", encoding="utf-8")
    with ix.updating(ix.index_path(notes)):
        code, out, err = run(capsys, "passport", str(notes), "--mode", "keyword")
    assert code == 0 and "another sfind is updating" in err and "passport.md" in out


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_a_merge_conflict_does_not_break_indexing_and_tracked_build_files_count(tmp_path, capsys):
    d = tmp_path / "repo"
    d.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *a],
                       cwd=d, env=env, capture_output=True, check=False)

    git("init", "-q")
    (d / "notes.md").write_text("base\n", encoding="utf-8")
    (d / "build").mkdir()
    (d / "build" / "release.md").write_text("release checklist", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "a")
    git("checkout", "-qb", "other")
    (d / "notes.md").write_text("other side\n", encoding="utf-8")
    git("commit", "-qam", "b")
    git("checkout", "-q", "-")
    (d / "notes.md").write_text("this side\n", encoding="utf-8")
    git("commit", "-qam", "c")
    git("merge", "-q", "other")
    (d / "node_modules").mkdir()
    (d / "node_modules" / "x.md").write_text("release", encoding="utf-8")
    found = sorted(p.relative_to(d).as_posix() for p in discover(d))
    assert found.count("notes.md") == 1 and "build/release.md" in found and "node_modules/x.md" not in found
    code, out, _ = run(capsys, "release", str(d), "--mode", "keyword")
    assert code == 0 and "release.md" in out


def test_a_zip_bomb_is_not_unpacked(tmp_path, monkeypatch):
    monkeypatch.setattr("semantic_find.text.ZIP_BUDGET", 10_000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "<w:document xmlns:w='w'>" + " " * 50_000 + "</w:document>")
    assert len(buf.getvalue()) < 2000, "small on disk"
    assert read_text(tmp_path / "bomb.docx", buf.getvalue()) is None


def test_a_glob_is_applied_before_the_candidates_are_cut(tmp_path, monkeypatch, capsys):
    d = tmp_path / "proj"
    d.mkdir()
    for i in range(12):
        (d / f"m{i}.py").write_text("widget widget widget", encoding="utf-8")
    (d / "guide.md").write_text("a widget", encoding="utf-8")
    monkeypatch.setattr(ix, "CANDIDATES", 5)
    code, out, _ = run(capsys, "widget", str(d), "--mode", "keyword", "-g", "*.md")
    assert code == 0 and "guide.md" in out


def test_a_download_whose_part_is_already_whole_finishes(tmp_path, monkeypatch, home):
    from test_embed import Server, pin

    from semantic_find.embed import download, is_downloaded

    src = make_model(tmp_path / "src")
    files = {f: (src / f).read_bytes() for f in ("config.json", "tokenizer.json", "model.safetensors")}
    hf = Server(files)
    pin(monkeypatch, files)
    monkeypatch.setenv("HF_ENDPOINT", hf.url)
    try:
        folder = home / "models" / "potion-base-8M"
        folder.mkdir(parents=True)
        (folder / "model.safetensors.part").write_bytes(files["model.safetensors"])  # interrupted while checking
        (folder / "tokenizer.json.part").write_bytes(files["tokenizer.json"] + b"extra")  # longer than it should be
        download("potion-base-8M")
        assert is_downloaded("potion-base-8M")
        assert not any("model.safetensors" in path for path, _ in hf.requests), "nothing to fetch: it was only checked"
    finally:
        hf.close()


def test_piped_output_keeps_file_names_whole(tmp_path, home):
    d = tmp_path / "docs"
    d.mkdir()
    name = "café 여권 ✓ notes.md"
    (d / name).write_text("passport", encoding="utf-8")
    env = {**os.environ, "SEMANTIC_FIND_HOME": str(home), "PYTHONIOENCODING": ""}
    out = subprocess.run([sys.executable, "-m", "semantic_find.cli", "passport", str(d), "--mode", "keyword", "-l", "-q"],
                         capture_output=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    path = out.stdout.decode("utf-8").strip()
    assert path.endswith(name) and os.path.exists(path)


@pytest.mark.parametrize("enc,text", [
    ("cp1252", "Crème brûlée recipe: bake at 180 degrees."), ("cp1252", "Café résumé naïve"),
    ("shift_jis", "パスポートの更新は早めに。"), ("euc_kr", "여권 갱신은 미리 하세요."), ("euc_kr", "여권 갱신"),
    ("gb18030", "数据库连接池的最大连接数设为二十。"), ("gb18030", "中文文档"), ("big5", "護照換發需要七個工作天。"),
])
def test_legacy_encodings_are_told_apart(enc, text):
    assert decode(text.encode(enc)) == text


def test_query_words_that_name_folders_stay_words(notes):
    (notes / "tests").mkdir()
    assert cli._split_search(["how", "tests", "work", str(notes)]) == ["search", "how", "tests", "work", "--in", str(notes)]


def test_no_update_with_another_model_is_explained(notes, model_path, tmp_path, capsys):
    run(capsys, "index", str(notes), "--model", str(model_path))
    other = make_model(tmp_path / "other")
    code, _, err = run(capsys, "passport", str(notes), "--model", str(other), "--no-update")
    assert code == 2 and "was built with" in err


def test_index_names_keep_case_where_the_file_system_does(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert ix.index_path(tmp_path / "Notes") != ix.index_path(tmp_path / "notes")
    monkeypatch.setattr(sys, "platform", "win32")
    assert ix.index_path(tmp_path / "Notes") == ix.index_path(tmp_path / "notes")


def test_ollama_listening_everywhere_is_reached_here():
    assert is_local("http://0.0.0.0:11434")


def test_one_chinese_character_is_found(tmp_path, capsys):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "note.md").write_text("我的车坏了，要去修。", encoding="utf-8")  # (not 车.md: the file name would match)
    code, out, _ = run(capsys, "车", str(d), "--mode", "keyword")
    assert code == 0 and "note.md" in out


def test_office_tabs_breaks_text_boxes_and_inline_cells(tmp_path):
    w = 'xmlns:w="w"'
    xml = (f'<w:document {w}><w:body><w:p><w:r><w:t>Name</w:t><w:tab/><w:t>Value</w:t><w:br/><w:t>Next</w:t></w:r>'
           f'<w:r><w:pict><w:p><w:r><w:t>In a box</w:t></w:r></w:p></w:pict></w:r></w:p></w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    text = read_text(tmp_path / "a.docx", buf.getvalue())
    assert "Name\tValue\nNext" in text and text.count("In a box") == 1
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="x"><sheetData><row><c t="inlineStr"><is><t>Typed in</t></is></c></row></sheetData></worksheet>')
    assert read_text(tmp_path / "b.xlsx", buf.getvalue()) == "Typed in"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("content.xml", '<o xmlns:text="t"><text:p>A<text:tab/>B<text:line-break/>C<text:s text:c="2"/>D</text:p></o>')
    assert read_text(tmp_path / "c.odt", buf.getvalue()) == "A\tB\nC  D"


def test_a_crafted_safetensors_file_is_refused(tmp_path):
    p = tmp_path / "m.safetensors"
    write_safetensors(p, {"embeddings": np.zeros((4, 2), np.float32)})
    raw = bytearray(p.read_bytes())
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + header_len])
    header["embeddings"]["shape"] = [400, 2]
    header["embeddings"]["data_offsets"] = [0, 3200]
    new = json.dumps(header).encode()
    p.write_bytes(len(new).to_bytes(8, "little") + new + raw[8 + header_len:])
    with pytest.raises(EmbedError, match="outside the file"):
        read_safetensors(p)
    p.write_bytes(b"\xff" * 16)
    with pytest.raises(EmbedError, match="not a safetensors file"):
        read_safetensors(p)


def test_a_model_folder_with_a_bad_tensor_is_an_error_not_a_crash(tmp_path):
    folder = make_model(tmp_path / "m")
    (folder / "model.safetensors").write_bytes(b"\x05\x00\x00\x00\x00\x00\x00\x00{bad}")
    with pytest.raises(EmbedError):
        StaticModel(folder)
