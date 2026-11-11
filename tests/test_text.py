import io
import json
import zipfile

from semantic_find.text import STOP, decode, passages, query_terms, read_text, terms


def test_terms_split_identifiers_and_pair_cjk_characters():
    assert terms("parseHttpRequest") == ["parsehttprequest", "parse", "http", "request"]
    assert terms("max_retry_count") == ["max", "retry", "count"], "as FTS5 stores them: it splits at _"
    assert terms("数据库连接") == ["数据", "据库", "库连", "连接"]
    assert terms("用 Python 写") == ["用", "python", "写"]
    assert terms("Café don't") == ["cafe", "don", "t"], "accents folded: cafe finds café"


def test_query_terms_drop_filler_unless_that_is_all_there_is():
    assert query_terms("how do I renew my passport") == ["renew", "passport"]
    assert query_terms("怎么配置数据库") == ["么配", "配置", "置数", "数据", "据库"]
    assert query_terms("the who") == ["the", "who"]
    assert "怎么" in STOP and "the" in STOP


def test_passages_follow_headings_paragraphs_and_limits():
    text = "# Intro\n\nfirst paragraph\n\n## Details\n\nsecond\nstill second\n\n" + "x " * 2000
    ps = passages(text, target=40, limit=300)
    assert ps[0].context == "Intro" and ps[0].line == 1 and "first paragraph" in ps[0].text
    second = next(p for p in ps if "still second" in p.text)
    assert second.context == "Details" and second.line == 5
    assert all(len(p.text) <= 300 for p in ps), "no passage is far over the limit"
    assert passages("# comment\ncode()", headings=False)[0].context == ""


def test_pages_are_counted_by_form_feeds():
    ps = passages("page one text\fpage two text\f\fpage four")
    assert [(p.page, p.text) for p in ps] == [(1, "page one text"), (2, "page two text"), (4, "page four")]


def test_one_enormous_line_is_cut():
    ps = passages("a" * 5000, limit=1500)
    assert len(ps) == 4 and sum(len(p.text) for p in ps) == 5000


def test_decoding_utf8_utf16_gb18030_and_binary():
    assert decode("héllo".encode()) == "héllo"
    assert decode("﻿bom".encode()) == "bom"
    assert decode("你好".encode("utf-16")) == "你好"
    assert decode("中文文档".encode("gb18030")) == "中文文档"
    assert decode(b"\x00\x01\x02binary") is None


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'


def test_office_documents_are_read_as_text(tmp_path):
    docx = _zip({"word/document.xml": f'<w:document {W}><w:body><w:p><w:r><w:t>Renew the </w:t></w:r><w:r><w:t>passport</w:t></w:r></w:p>'
                                      f'<w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p></w:body></w:document>'})
    assert read_text(tmp_path / "a.docx", docx) == "Renew the passport\nSecond paragraph"
    pptx = _zip({"ppt/slides/slide2.xml": f'<p:sld xmlns:p="p" {A}><a:p><a:r><a:t>Two</a:t></a:r></a:p></p:sld>',
                 "ppt/slides/slide10.xml": f'<p:sld xmlns:p="p" {A}><a:p><a:r><a:t>Ten</a:t></a:r></a:p></p:sld>'})
    assert read_text(tmp_path / "s.pptx", pptx) == "# Slide 1\nTwo\n\n# Slide 2\nTen", "slides in their order, not the zip's"
    xlsx = _zip({"xl/sharedStrings.xml": '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Budget</t></si></sst>'})
    assert read_text(tmp_path / "b.xlsx", xlsx) == "Budget"
    odt = _zip({"content.xml": '<o:document xmlns:o="o" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
                               '<text:h>Title</text:h><text:p>Body <text:span>text</text:span></text:p></o:document>'})
    assert read_text(tmp_path / "d.odt", odt) == "Title\nBody text"
    assert read_text(tmp_path / "broken.docx", b"not a zip") is None


def test_html_notebooks_and_binaries(tmp_path):
    html = "<html><head><style>p{}</style><script>var passport=1</script></head><body><h2>Visa</h2><p>Apply &amp; wait</p></body></html>"
    assert read_text(tmp_path / "p.html", html.encode()) == "## Visa\n\nApply & wait"
    nb = json.dumps({"cells": [{"source": ["# Notes\n", "about dough"]}, {"source": "print(1)"}]})
    assert read_text(tmp_path / "n.ipynb", nb.encode()) == "# Notes\nabout dough\n\nprint(1)"
    assert read_text(tmp_path / "x.png", b"passport") is None
    assert read_text(tmp_path / "x.bin.txt", b"\x00\x00\x00") is None


def test_matched_character_pairs_are_shown_as_words():
    from semantic_find.text import join_pairs

    assert join_pairs("数据库连接池怎么配置", ["连接", "接池", "配置"]) == ["连接池", "配置"]
    assert join_pairs("renew passport", ["passport"]) == ["passport"]
    assert join_pairs("护照和签证", ["护照", "签证", "visa"]) == ["护照", "签证", "visa"]


def test_pdf_pages_are_read_and_counted(tmp_path):
    import pytest

    pytest.importorskip("pypdf")
    from pdfmaker import make_pdf

    text = read_text(tmp_path / "guide.pdf", make_pdf(["Renew the passport early", "Visa rules for Japan"]))
    assert "Renew the passport early" in text and "\f" in text
    assert [(p.page, p.text) for p in passages(text)] == [(1, "Renew the passport early"), (2, "Visa rules for Japan")]
    assert read_text(tmp_path / "broken.pdf", b"%PDF-1.4 not really") is None, "a broken PDF is skipped, not an error"
