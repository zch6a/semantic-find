"""Turning files into searchable passages: reading the text out of a file, cutting it into passages a
person can read in one look, and the words a keyword search matches on.

Keyword terms work for Chinese, Japanese and Korean as well as for English: those scripts do not put
spaces between words, so a run of their characters is indexed as overlapping pairs ("数据库" gives
"数据" and "据库"). A two-character word then matches wherever it appears, with no dictionary.
"""

from __future__ import annotations

import html.parser
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

CJK = "぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"
_TOKEN = re.compile(rf"[{CJK}]+|[^\W_{CJK}]+(?:[_'’][^\W_{CJK}]+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_CJK_RUN = re.compile(rf"^[{CJK}]+$")


def terms(text: str) -> list[str]:
    """Keyword terms: words in lower case, identifiers also split into their parts ("parseHttpRequest"
    gives "parsehttprequest", "parse", "http", "request"), and CJK text as character pairs."""
    out: list[str] = []
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        if _CJK_RUN.match(tok):
            if len(tok) > 1:
                out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
            else:
                out.append(tok)
            continue
        low = tok.lower()
        out.append(low)
        parts = [p.lower() for piece in re.split(r"[_'’]", tok) for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            out.extend(parts)
    return out


# Words that say nothing about what is wanted ("how do I ..."), left out of a query's keywords.
STOP = set(["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "should", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "about", "after", "again", "all", "also", "am", "any", "because", "before", "being", "both", "each", "few", "further", "here", "just", "more", "most", "other", "over", "own", "same", "some", "such", "too", "under", "until", "up", "very", "via", "vs", "off", "out", "only", "once", "while"]) | {
    "怎么", "什么", "如何", "为什", "么样", "哪里", "哪个", "一个", "我的", "我们", "你们", "他们", "可以", "应该", "需要", "是否",
    "有没", "这个", "那个", "这些", "那些", "怎样", "的", "了", "是", "在", "和", "与", "或", "吗", "呢", "吧", "啊"}


def query_terms(query: str) -> list[str]:
    """A query's keyword terms without the filler, unless filler is all there is ("the who")."""
    all_terms = list(dict.fromkeys(terms(query)))
    kept = [t for t in all_terms if t not in STOP]
    return kept or all_terms


def join_pairs(query: str, found: list[str]) -> list[str]:
    """The matched words as a person would say them: character pairs that overlap in the query are
    joined back ("连接" and "接池" from "连接池" are shown as "连接池"); other terms stay as they are."""
    have = set(found)
    out: list[str] = []
    used: set[str] = set()
    for m in re.finditer(rf"[{CJK}]+", query):
        run = m.group(0)
        i = 0
        while i < len(run) - 1:
            if run[i:i + 2] not in have:
                i += 1
                continue
            j = i
            while j < len(run) - 1 and run[j:j + 2] in have:
                used.add(run[j:j + 2])
                j += 1
            out.append(run[i:j + 1])
            i = j
    return list(dict.fromkeys([*out, *(t for t in found if t not in used)]))


@dataclass
class Passage:
    line: int  # first line, from 1
    text: str  # as shown
    context: str = ""  # the heading it is under, given to the embedding model with the text
    page: int | None = None  # for paged documents (PDF)


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


# Formats whose "# " lines are headings (Markdown, and what is read out of HTML, Office and EPUB
# files); in code and plain text a "# " line is a comment or just text.
HEADED = {".md", ".markdown", ".mdx", ".html", ".htm", ".xhtml", ".docx", ".pptx", ".odt", ".odp", ".epub"}


def passages(text: str, target: int = 700, limit: int = 1500, headings: bool = True) -> list[Passage]:
    """Paragraphs, merged up to about `target` characters and never much over `limit`. With `headings`,
    a Markdown heading starts a new passage and is remembered as the context of the ones under it.
    Form feeds (page breaks, as PDF text has) count pages."""
    out: list[Passage] = []
    heading, page = "", 1 if "\f" in text else None
    buf: list[str] = []
    buf_line, buf_page = 1, page

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip("\n")
        if body.strip():
            out.append(Passage(buf_line, body, heading, buf_page))
        buf = []

    def pieces():
        """(line number, text) for each line, with a form feed (which may come in the middle of a
        line, as in PDF text) ending the page: (line number, None)."""
        for i, raw in enumerate(text.split("\n"), start=1):
            for j, seg in enumerate(raw.split("\f")):
                if j:
                    yield i, None
                yield i, seg

    size = 0
    for i, raw in pieces():
        if raw is None:
            flush()
            page = (page or 1) + 1
            size = 0
            continue
        line = raw.rstrip("\r")
        h = HEADING.match(line) if headings else None
        if h:
            flush()
            size = 0
            heading = h.group(2).strip()
        blank = not line.strip()
        if blank and size >= target:
            flush()
            size = 0
            continue
        while len(line) > limit:  # one enormous line (minified code, a CSV without breaks)
            if buf:
                flush()
            buf_line, buf_page = i, page
            buf.append(line[:limit])
            flush()
            line = line[limit:]
            size = 0
        if not buf:
            if blank:
                continue
            buf_line, buf_page = i, page
        if size + len(line) > limit and buf:
            flush()
            buf_line, buf_page, size = i, page, 0
        buf.append(line)
        size += len(line) + 1
    flush()
    return out


# ---------------------------------------------------------------- reading files

TEXT_LIMIT = 20 * 1024 * 1024
BINARY = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff", ".psd", ".heic", ".svgz",
    ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".zst", ".tar", ".jar", ".whl", ".apk", ".dmg", ".iso",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib", ".obj", ".pdb", ".class", ".pyc", ".pyd", ".wasm", ".bin",
    ".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".mov", ".avi", ".mkv", ".webm",
    ".ttf", ".otf", ".woff", ".woff2", ".eot", ".sqlite", ".db", ".npy", ".npz", ".pt", ".pth", ".onnx",
    ".safetensors", ".gguf", ".parquet", ".lock", ".doc", ".xls", ".ppt",
}


def decode(data: bytes) -> str | None:
    """Text from bytes, or None for a binary file. UTF-8 first, then UTF-16 with a byte-order mark, then
    GB18030, which many older Chinese documents use."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    if b"\x00" in data[:8192]:
        return None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


class _HTMLText(html.parser.HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "template"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "template"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _html(text: str) -> str:
    p = _HTMLText()
    p.feed(text)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(p.parts)).strip()


def _xml_text(xml: bytes, paragraph: str, run: str) -> str:
    """The text of an Office Open XML or OpenDocument part: one line per paragraph."""
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    lines = []
    for p in root.iter():
        if p.tag.endswith("}" + paragraph):
            lines.append("".join(t.text or "" for t in p.iter() if t.tag.endswith("}" + run)))
    return "\n".join(lines)


def _office(path: Path, data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        ext = path.suffix.lower()
        if ext == ".docx":
            parts = ["word/document.xml"] + sorted(n for n in names if re.match(r"word/(footnotes|endnotes)\.xml$", n))
            return "\n\n".join(_xml_text(z.read(n), "p", "t") for n in parts if n in names)
        if ext == ".pptx":
            slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
            return "\n\n".join(f"# Slide {i}\n" + _xml_text(z.read(n), "p", "t") for i, n in enumerate(slides, start=1))
        if ext == ".xlsx":
            return _xml_text(z.read("xl/sharedStrings.xml"), "si", "t") if "xl/sharedStrings.xml" in names else ""
        if ext in (".odt", ".odp", ".ods"):
            return _odf(z.read("content.xml"))
        if ext == ".epub":
            docs = sorted(n for n in names if n.endswith((".xhtml", ".html", ".htm")))
            return "\n\n".join(_html(z.read(n).decode("utf-8", "replace")) for n in docs)
    return ""


def _odf(xml: bytes) -> str:
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    lines = []
    for el in root.iter():
        if el.tag.endswith("}p") or el.tag.endswith("}h"):
            lines.append("".join(el.itertext()))
    return "\n".join(lines)


def _pdf(data: bytes) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None  # reading PDF needs the optional extra: pip install "semantic-find[pdf]"
    reader = PdfReader(io.BytesIO(data))
    return "\f".join((page.extract_text() or "") for page in reader.pages)


def _notebook(text: str) -> str:
    nb = json.loads(text)
    cells = []
    for c in nb.get("cells", []):
        src = c.get("source", "")
        cells.append("".join(src) if isinstance(src, list) else str(src))
    return "\n\n".join(cells)


DOCUMENTS = {".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".epub", ".pdf"}


def read_text(path: Path, data: bytes | None = None) -> str | None:
    """The searchable text of a file, or None when it has none (binary, too big, or a PDF without the
    PDF extra). Office documents, OpenDocument, EPUB, HTML and Jupyter notebooks are read as their text."""
    ext = path.suffix.lower()
    if ext in BINARY:
        return None
    if data is None:
        data = path.read_bytes()
    try:
        if ext == ".pdf":
            return _pdf(data)
        if ext in DOCUMENTS:
            return _office(path, data)
    except (zipfile.BadZipFile, KeyError, ValueError, SyntaxError):
        return None
    if len(data) > TEXT_LIMIT:
        return None
    text = decode(data)
    if text is None:
        return None
    if ext in (".html", ".htm", ".xhtml"):
        return _html(text)
    if ext == ".ipynb":
        try:
            return _notebook(text)
        except (ValueError, AttributeError):
            return text
    return text
