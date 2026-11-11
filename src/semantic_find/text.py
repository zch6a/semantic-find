"""Turning files into searchable passages: reading the text out of a file, cutting it into passages a
person can read in one look, and the words a keyword search matches on.

Keyword terms work for Chinese, Japanese and Korean as well as for English: those scripts do not put
spaces between words, so a run of their characters is indexed as overlapping pairs ("数据库" gives
"数据" and "据库"). A two-character word then matches wherever it appears, with no dictionary.
"""

from __future__ import annotations

import functools
import html.parser
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"  # kana, CJK ideographs, Hangul
_TOKEN = re.compile(rf"[{CJK}]+|[^\W_{CJK}]+(?:[_'’][^\W_{CJK}]+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_JOINERS = re.compile(r"[_'’]")
_CJK_RUN = re.compile(rf"^[{CJK}]+$")


@functools.lru_cache(maxsize=200_000)
def _word_terms(tok: str) -> tuple[str, ...]:
    """The terms of one word. Words repeat all the time (in code above all), so each is worked out
    once: this made indexing about twice as fast."""
    if _CJK_RUN.match(tok):
        return tuple(tok[i:i + 2] for i in range(len(tok) - 1)) if len(tok) > 1 else (tok,)
    parts = [p.lower() for piece in _JOINERS.split(tok) for p in _CAMEL.findall(piece)]
    return (tok.lower(), *parts) if len(parts) > 1 else (tok.lower(),)


def index_terms(text: str) -> list[str]:
    """The terms a passage is indexed under: its keyword terms, and for CJK text each character too, so
    that a one-character query ("车") finds "汽车" and "车坏了"."""
    out = terms(text)
    out.extend(c for run in re.findall(rf"[{CJK}]{{2,}}", text) for c in run)
    return out


def terms(text: str) -> list[str]:
    """Keyword terms: words in lower case, identifiers also split into their parts ("parseHttpRequest"
    gives "parsehttprequest", "parse", "http", "request"), and CJK text as character pairs."""
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        out.extend(_word_terms(tok))
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


LEGACY = ("gb18030", "big5", "shift_jis", "euc_kr", "cp1252")


def _common(enc: str, ch: str) -> bool:
    """Whether a character is an everyday one in the script that `enc` is for. Text decoded with the
    wrong encoding is full of rare characters (cp1252 "Crème" read as GB18030 is "Cr鑝e"), so the share of
    everyday ones tells the right decoding from the wrong ones."""
    o = ord(ch)
    if enc == "gb18030":
        try:
            b = ch.encode("gb2312")
        except UnicodeEncodeError:
            return False
        return len(b) == 2 and (0xB0 <= b[0] <= 0xD7 or b[0] in (0xA1, 0xA3))  # the 3,755 common hanzi; punctuation
    if enc == "big5":
        try:
            b = ch.encode("big5")
        except UnicodeEncodeError:
            return False
        return len(b) == 2 and (0xA4 <= b[0] <= 0xC6 or b[0] == 0xA1)  # frequent hanzi; punctuation
    if enc == "shift_jis":
        if 0x3040 <= o <= 0x30FF or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF:
            return True  # kana, punctuation, full-width forms
        try:
            b = ch.encode("shift_jis")
        except UnicodeEncodeError:
            return False
        return len(b) == 2 and 0x88 <= b[0] <= 0x9F  # JIS level-1 kanji
    if enc == "euc_kr":
        return 0xAC00 <= o <= 0xD7A3 or 0x3000 <= o <= 0x303F  # Hangul syllables
    if enc == "cp1252":
        return (ch.isalpha() and o < 0x250) or ch in "‘’“”–—…€•°±·×÷§©®™«»¿¡"
    return False


def decode(data: bytes) -> str | None:
    """Text from bytes, or None for a binary file. UTF-16 with a byte-order mark, then UTF-8; failing
    those, the legacy encoding (GB18030, Big5, Shift-JIS, EUC-KR or Windows-1252) whose reading is made
    most of everyday characters."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    best: tuple[float, str] | None = None
    sample = data[:65536]
    for enc in LEGACY:
        try:
            text = sample.decode(enc)
        except UnicodeDecodeError:
            try:  # the sample may end in the middle of a character
                text = sample[:-3].decode(enc)
            except UnicodeDecodeError:
                continue
        wide = [c for c in text if ord(c) > 0x7F]
        score = sum(_common(enc, c) for c in wide) / len(wide) if wide else 1.0
        if enc == "euc_kr" and wide and len(re.findall(r"[^\x00-\x7f] [^\x00-\x7f]", text)) * 8 >= len(wide):
            score += 0.01  # Korean puts spaces between words; Chinese, read from the same bytes, does not
        if best is None or score > best[0]:
            best = (score, enc)
    if best is None:
        return None
    return data.decode(best[1], errors="replace")


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


ZIP_BUDGET = 200 * 1024 * 1024  # bytes unpacked from one document, at most: a "zip bomb" is skipped, not unpacked
ZIP_MEMBERS = 5000


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ooxml(xml: bytes, paragraph: str = "p") -> str:
    """The text of an Office Open XML part: one line per paragraph, tabs and line breaks kept. A paragraph
    inside another (in a text box) is its own line, not repeated in the outer one."""
    from xml.etree import ElementTree

    lines: list[str] = []

    def walk(el, buf):
        name = _local(el.tag)
        if name == paragraph:
            mine: list[str] = []
            for c in el:
                walk(c, mine)
            lines.append("".join(mine))
            return
        if buf is not None:
            if name == "t":
                buf.append(el.text or "")
            elif name == "tab":
                buf.append("\t")
            elif name in ("br", "cr"):
                buf.append("\n")
        for c in el:
            walk(c, buf)

    walk(ElementTree.fromstring(xml), None)
    return "\n".join(lines)


def _odf(xml: bytes) -> str:
    """The text of an OpenDocument content.xml: paragraphs and headings, with tabs, line breaks and runs
    of spaces as they are written."""
    from xml.etree import ElementTree

    lines: list[str] = []

    def walk(el, buf):
        name = _local(el.tag)
        if name in ("p", "h"):
            mine: list[str] = [el.text or ""]
            for c in el:
                walk(c, mine)
            lines.append("".join(mine))
        else:
            if buf is not None:
                if name == "tab":
                    buf.append("\t")
                elif name == "line-break":
                    buf.append("\n")
                elif name == "s":
                    count = next((v for k, v in el.attrib.items() if _local(k) == "c"), "1")
                    buf.append(" " * (int(count) if count.isdigit() else 1))
                elif el.text:
                    buf.append(el.text)
            for c in el:
                walk(c, buf)
        if buf is not None and el.tail:
            buf.append(el.tail)

    walk(ElementTree.fromstring(xml), None)
    return "\n".join(lines)


def _office(path: Path, data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = {i.filename: i for i in z.infolist()}
        if len(infos) > ZIP_MEMBERS:
            raise ValueError("too many parts")
        spent = 0

        def read(name: str) -> bytes:
            nonlocal spent
            spent += infos[name].file_size
            if spent > ZIP_BUDGET:
                raise ValueError("too large unpacked")  # never unpacked: the declared size is checked first
            with z.open(name) as f:
                out = f.read(infos[name].file_size + 1)
            if len(out) > infos[name].file_size:
                raise ValueError("larger than it says")
            return out

        names = list(infos)
        ext = path.suffix.lower()
        if ext == ".docx":
            parts = ["word/document.xml"] + sorted(n for n in names if re.match(r"word/(footnotes|endnotes)\.xml$", n))
            return "\n\n".join(_ooxml(read(n)) for n in parts if n in infos)
        if ext == ".pptx":
            slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
            return "\n\n".join(f"# Slide {i}\n" + _ooxml(read(n)) for i, n in enumerate(slides, start=1))
        if ext == ".xlsx":
            out = [_ooxml(read("xl/sharedStrings.xml"), "si")] if "xl/sharedStrings.xml" in infos else []
            sheets = sorted(n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
            out += [_ooxml(read(n), "is") for n in sheets]  # text typed into cells directly ("inline strings")
            return "\n".join(x for x in out if x.strip())
        if ext in (".odt", ".odp", ".ods"):
            return _odf(read("content.xml"))
        if ext == ".epub":
            docs = sorted(n for n in names if n.endswith((".xhtml", ".html", ".htm")))
            return "\n\n".join(_html(read(n).decode("utf-8", "replace")) for n in docs)
    return ""


def _pdf(data: bytes) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None  # reading PDF needs the optional extra: pip install "semantic-find[pdf]"
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\f".join((page.extract_text() or "") for page in reader.pages)
    except Exception:  # noqa: BLE001 - pypdf raises many kinds of error for a damaged or encrypted file: no text
        return None


def _notebook(text: str) -> str:
    nb = json.loads(text)
    cells = []
    for c in (nb.get("cells") or []) if isinstance(nb, dict) else []:
        src = c.get("source", "") if isinstance(c, dict) else ""
        cells.append("".join(map(str, src)) if isinstance(src, list) else str(src or ""))
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
