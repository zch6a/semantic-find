"""sfind: search files by what they mean, on this machine.

    sfind "how do I renew my passport" ~/Documents
    sfind "数据库连接池怎么配置" .
    sfind index ~/notes          build or refresh the index without searching
    sfind models download        fetch the default model (once)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

from . import __version__
from .embed import DEFAULT_MODEL, MODELS, EmbedError, download, is_downloaded, load, model_dir
from .index import SCORE_FLOOR, Busy, Hit, Index, index_path

COMMANDS = {"index", "status", "forget", "models", "search"}


def _err(msg: str) -> None:
    print(f"sfind: {msg}", file=sys.stderr)


def _model_name(args, idx: Index | None = None) -> str:
    return args.model or os.environ.get("SFIND_MODEL") or (idx.meta("model") if idx else None) or DEFAULT_MODEL


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


_last_tenth: dict[str, int] = {}


def _progress_bar(file: str, done: int, total: int) -> None:
    """On a terminal, one line that updates; elsewhere (a log, a CI job, a background process), a line
    at each tenth, so that a slow download is never silent."""
    if not total:
        return
    if sys.stderr.isatty():
        print(f"\r  {file}: {done / 1048576:6.1f} of {total / 1048576:.1f} MB", end="", file=sys.stderr, flush=True)
        return
    tenth = done * 10 // total
    if tenth > _last_tenth.get(file, -1) and total > 4 * 1048576:
        _last_tenth[file] = tenth
        print(f"  {file}: {tenth * 10}% of {total / 1048576:.0f} MB", file=sys.stderr, flush=True)


def _n(count: int, word: str) -> str:
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def get_model(name: str, *, ask: bool, needed: bool = False):
    """The model, or None when it is a built-in one not downloaded yet and the user did not agree to
    download it now (searching then matches keywords only)."""
    if name in MODELS and not is_downloaded(name):
        info = MODELS[name]
        if not ask or not _interactive():
            if needed:
                raise EmbedError(f"searching by meaning needs the model {name}, which is not downloaded yet: "
                                 f"`sfind models download` fetches it ({info.size_mb} MB, once)")
            _err(f"keywords only: the model {name} is not downloaded. `sfind models download` fetches it "
                 f"({info.size_mb} MB, once), then search by meaning works offline.")
            return None
        answer = input(f"sfind: to search by meaning, the model {name} ({info.languages}, {info.size_mb} MB) is downloaded once "
                       f"from {os.environ.get('HF_ENDPOINT') or 'huggingface.co'}. Download it now? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            _err("searching by keywords only")
            return None
        download(name, progress=_progress_bar)
        print(file=sys.stderr)
    return load(name)


def _open(path: str) -> Index:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such file or folder: {path}")
    return Index(p)


def _update(idx: Index, model, name: str, quiet: bool) -> None:
    def progress(msg: str) -> None:
        if not quiet:
            print(f"\rsfind: {msg}...", end="", file=sys.stderr, flush=True)

    try:
        st = idx.update(model, name if model is not None else None, progress=progress if _interactive() else None)
    except Busy:
        _err(f"another sfind is updating the index of {idx.root}; searching it as it is")
        return
    if not quiet and (st.read or st.removed or st.embedded):
        print("\r", end="", file=sys.stderr)
        parts = [f"{st.read} file{'s' if st.read != 1 else ''} read"]
        if st.removed:
            parts.append(f"{st.removed} gone")
        if st.embedded:
            parts.append(f"{st.embedded} passage{'s' if st.embedded != 1 else ''} embedded")
        _err(f"index of {idx.root} updated: {', '.join(parts)} in {st.seconds:.1f} s")


# ---------------------------------------------------------------- output

def _color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def snippet(text: str, words: list[str], width: int = 240, heading: str = "") -> str:
    """A few lines of the passage, from around the first word that matched. The passage's own heading
    line is shown as "Heading — text", not glued to the sentence after it."""
    def show(line: str) -> str:
        m = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        return f"{m.group(1)} —" if heading and m and m.group(1).strip() == heading else line

    lines = [show(x.strip()) for x in text.splitlines() if x.strip()]
    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    low = flat.lower()
    first = min((low.find(w) for w in words if low.find(w) >= 0), default=0)
    start = max(0, first - width // 3)
    if start:
        space = flat.find(" ", start)
        start = space + 1 if 0 <= space < first else start
    # Cut by width on screen, not by characters: a Chinese character takes two columns.
    end, cols = start, 0
    while end < len(flat) and cols < width:
        cols += 2 if unicodedata.east_asian_width(flat[end]) in ("W", "F") else 1
        end += 1
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


def highlight(text: str, words: list[str]) -> str:
    if not words or not _color():
        return text
    pattern = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)), re.I)
    return pattern.sub(lambda m: f"\033[1;33m{m.group(0)}\033[0m", text)


def _shown(root: Path, rel: str) -> str:
    """A path as it is shown: relative to the current folder when that is shorter, as grep shows it."""
    full = root / rel if root.is_dir() else root
    try:
        shown = os.path.relpath(full)
        return shown if not shown.startswith("..") or len(shown) < len(str(full)) else str(full)
    except ValueError:  # another drive on Windows
        return str(full)


def _where(root: Path, hit: Hit) -> str:
    shown = _shown(root, hit.path)
    return f"{shown} (page {hit.page})" if hit.page else f"{shown}:{hit.line}"


def why(hit: Hit) -> str:
    parts = []
    if hit.words:
        parts.append("words: " + ", ".join(hit.words[:6]))
    if hit.similarity is not None:
        parts.append(f"meaning {hit.similarity:.2f}")
    if hit.more:
        parts.append(f"{hit.more} more in this file")
    if hit.same:
        n = len(hit.same)
        parts.append(f"the same text in {n} other file{'s' if n != 1 else ''}" + (": " + ", ".join(hit.same) if n <= 2 else ""))
    return " · ".join(parts)


# ---------------------------------------------------------------- commands

def cmd_search(args) -> int:
    query = " ".join(args.query).strip()
    if not query:
        _err("what should be searched for?")
        return 2
    if args.top < 1:
        _err("-k must be at least 1")
        return 2
    if len(query) > MAX_QUERY:
        _err(f"the query is {len(query):,} characters long; only the first {MAX_QUERY:,} are used")
        query = query[:MAX_QUERY]
    args.paths = _distinct_roots(args.paths)
    if args.mode == "keyword":
        model = name = None
    else:
        first = _open(args.paths[0])
        name = _model_name(args, first)
        first.close()
        model = get_model(name, ask=True, needed=args.mode == "meaning")
        if model is None and args.mode == "meaning":
            return 2
    indexes = [_open(path) for path in args.paths]
    results: list[tuple[Path, Hit]] = []
    try:
        for idx in indexes:
            if not args.no_update:
                _update(idx, model, name, args.quiet)
            elif model is not None and idx.meta("model") and idx.meta("model") != name:
                raise RuntimeError(f"the index of {idx.root} was built with {idx.meta('model')}, not {name}: "
                                   f"search without --no-update to embed it again, or pass --model {idx.meta('model')}")
        mode = args.mode if model is not None else "keyword"
        vector = best = None
        if model is not None and mode != "keyword":
            vector = model.encode([query])[0]
            if len(indexes) > 1:  # one scale for all: each folder scored against the best match in any of them
                best = max((i.best_similarity(vector) for i in indexes if i.counts()["embedded"]), default=None)
        for idx in indexes:
            for hit in idx.search(query, model, k=args.top, mode=mode, glob=args.glob, per_file=not args.all,
                                  query_vector=vector, best=best):
                results.append((idx.root, hit))
    finally:
        for idx in indexes:
            idx.close()
    if any(not h.weak for _, h in results):  # strong results from one folder: the weak ones from another go
        results = [(r, h) for r, h in results if not h.weak]
    if len(indexes) > 1 and results and mode != "keyword":
        top = max(h.score for _, h in results)
        results = [(r, h) for r, h in results if h.score >= SCORE_FLOOR * top or h.full]
    results.sort(key=lambda r: r[1].score, reverse=True)
    results = results[:args.top]
    found = bool(results) and not all(h.weak for _, h in results)
    if args.json:
        print(json.dumps([{"path": str(root / h.path if root.is_dir() else root), "line": h.line, "page": h.page, "text": h.text,
                           "score": round(h.score, 6), "words": h.words,
                           "similarity": None if h.similarity is None else round(h.similarity, 4), "more": h.more,
                           "same": [str(root / p) for p in h.same], "weak": h.weak}
                          for root, h in results], ensure_ascii=False, indent=2))
        return 0 if found else 1
    if args.files:  # each matching file once, copies of a passage included; weak matches are not matches
        for shown in dict.fromkeys(_shown(root, rel) for root, h in results if not h.weak for rel in (h.path, *h.same)):
            print(shown)
        return 0 if found else 1
    if not results:
        _err("nothing found")
        return 1
    if not found:
        _err("no strong match; the closest passages, which may not be what you are looking for:")
    color = _color()
    for root, h in results:
        where = _where(root, h)
        print(f"\033[1;36m{where}\033[0m" if color else where)
        words = sorted(h.words, key=len, reverse=True)
        print("    " + highlight(snippet(h.text, words, heading=h.context), words))
        detail = why(h)
        if detail:
            print(f"    \033[2m({detail})\033[0m" if color else f"    ({detail})")
    return 0 if found else 1


MAX_QUERY = 2000  # characters: a query is a question, not a document


def _distinct_roots(paths: list[str]) -> list[str]:
    """The folders to search, each once: the same folder given twice, or a folder inside another one
    also given, would list every result twice."""
    resolved = []
    for p in paths:
        r = Path(p).expanduser().resolve()
        if r not in [x for x, _ in resolved]:
            resolved.append((r, p))
    keep = [(r, p) for r, p in resolved if not any(o != r and o.is_dir() and r.is_relative_to(o) for o, _ in resolved)]
    return [p for _, p in keep]


def cmd_index(args) -> int:
    for path in args.paths:
        idx = _open(path)
        try:
            name = _model_name(args, idx)
            model = None if args.keyword_only else get_model(name, ask=True)
            _update(idx, model, name, quiet=False)
            c = idx.counts()
            _err(f"{idx.root}: {_n(c['files'], 'file')}, {_n(c['passages'], 'passage')}, {c['embedded']:,} with vectors")
        finally:
            idx.close()
    return 0


def cmd_status(args) -> int:
    for path in args.paths:
        p = Path(path).expanduser().resolve()
        db = index_path(p)
        if not db.exists():
            print(f"{p}: not indexed yet")
            continue
        idx = Index(p, db)
        try:
            c = idx.counts()
            size = db.stat().st_size / 1048576
            print(f"{p}\n  {_n(c['files'], 'file')}, {_n(c['passages'], 'passage')}, {c['embedded']:,} with vectors ({idx.meta('model') or 'no model'})\n"
                  f"  index: {db} ({size:.1f} MB)")
        finally:
            idx.close()
    return 0


def cmd_forget(args) -> int:
    for path in args.paths:
        p = Path(path).expanduser().resolve()
        db = index_path(p)
        companions = [db.with_name(db.name + suffix) for suffix in ("-wal", "-shm", ".lock")]
        companions += [db.with_name(db.stem + suffix) for suffix in (".vectors.npy", ".vectors-ids.npy", ".vectors-version")]
        gone = [f for f in (db, *companions) if f.exists()]
        for f in gone:
            f.unlink()
        print(f"{p}: index deleted" if gone else f"{p}: there was no index")
    return 0


def cmd_models(args) -> int:
    if args.action == "download":
        name = args.name or DEFAULT_MODEL
        if name not in MODELS:
            _err(f"unknown model \"{name}\"; built in: {', '.join(MODELS)}")
            return 2
        print(f"downloading {name} ({MODELS[name].size_mb} MB) from {os.environ.get('HF_ENDPOINT') or 'https://huggingface.co'}", file=sys.stderr)
        path = download(name, progress=_progress_bar)
        print(f"\n{name} is in {path}", file=sys.stderr)
        return 0
    for m in MODELS.values():
        mark = "downloaded" if is_downloaded(m.name) else f"{m.size_mb} MB"
        default = " (default)" if m.name == DEFAULT_MODEL else ""
        print(f"{m.name}{default}\n    {m.languages}; {m.note}; {mark}")
        if is_downloaded(m.name):
            print(f"    {model_dir(m.name)}")
    print("ollama:<model>\n    any Ollama embedding model on this machine, such as ollama:bge-m3 or ollama:qwen3-embedding")
    return 0


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sfind", description="Search files by what they mean, on this machine.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="examples:\n"
                                       '  sfind "how do I renew my passport" ~/Documents\n'
                                       '  sfind "数据库连接池怎么配置" .\n'
                                       '  sfind "retry with backoff" src -g "*.py"\n'
                                       "  sfind models download          fetch the default model (once)\n\n"
                                       "exit codes: 0 found, 1 nothing found, 2 error")
    p.add_argument("--version", action="version", version=f"sfind {__version__}")
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument("--model", help=f"a built-in model, ollama:<name>, or a Model2Vec folder (default {DEFAULT_MODEL}, or SFIND_MODEL)")

    s = sub.add_parser("search", help="search (the default: `sfind QUERY [PATH...]`)")
    s.add_argument("query", nargs="+", help="what to look for, in your own words")
    s.add_argument("--in", dest="paths", nargs="+", default=None, metavar="PATH", help="folders or files to search (default: the current folder)")
    s.add_argument("-k", "--top", type=int, default=10, help="how many results (default 10)")
    s.add_argument("-g", "--glob", help="only files matching this pattern, such as \"*.md\" or \"docs/*\"")
    s.add_argument("--mode", choices=["hybrid", "meaning", "keyword"], default="hybrid", help="hybrid (default), by meaning only, or by keywords only")
    s.add_argument("-a", "--all", action="store_true", help="every matching passage, not just the best one per file")
    s.add_argument("-l", "--files", action="store_true", help="print only the paths of matching files")
    s.add_argument("--json", action="store_true", help="results as JSON")
    s.add_argument("--no-update", action="store_true", help="search the index as it is, without looking for changed files")
    s.add_argument("-q", "--quiet", action="store_true", help="no messages about the index")
    common(s)

    i = sub.add_parser("index", help="build or bring up to date the index of folders, without searching")
    i.add_argument("paths", nargs="*", default=["."])
    i.add_argument("--keyword-only", action="store_true", help="no vectors: keyword search only, no model needed")
    common(i)

    st = sub.add_parser("status", help="what is indexed, and where the index is")
    st.add_argument("paths", nargs="*", default=["."])

    f = sub.add_parser("forget", help="delete the index of folders (the files themselves are not touched)")
    f.add_argument("paths", nargs="*", default=["."])

    m = sub.add_parser("models", help="list the models, or download one")
    m.add_argument("action", nargs="?", choices=["list", "download"], default="list")
    m.add_argument("name", nargs="?", help=f"the model to download (default {DEFAULT_MODEL})")
    return p


def _looks_like_path(a: str) -> bool:
    """An existing file or folder, or one that would be in an existing folder ("~/notes/missing"),
    which is then reported as missing rather than searched for as words."""
    p = Path(a).expanduser()
    if p.exists():
        return True
    return ("/" in a or "\\" in a or a.startswith("~")) and p.parent != p and p.parent.exists() and str(p.parent) not in ("", ".")


def _split_search(argv: list[str]) -> list[str]:
    """`sfind QUERY [PATH...]`: the paths are the arguments at the end that are (or would be) paths, so a
    query word that happens to name a folder ("how tests work") stays a word. `sfind search ... --in
    PATH` is the explicit form."""
    if not argv or argv[0] in COMMANDS or argv[0] in ("-h", "--help", "--version"):
        return argv
    takes_value = {"-k", "--top", "-g", "--glob", "--mode", "--model", "--in"}
    plain, opts = [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("-") and len(a) > 1:
            opts.append(a)
            if a in takes_value and i + 1 < len(argv):
                opts.append(argv[i + 1])
                i += 1
        else:
            plain.append(a)
        i += 1
    n_paths = 0
    while n_paths < len(plain) - 1 and _looks_like_path(plain[len(plain) - 1 - n_paths]):
        n_paths += 1  # (the first word is always the query)
    words, paths = plain[:len(plain) - n_paths], plain[len(plain) - n_paths:]
    return ["search", *words, *opts, *(["--in", *paths] if paths else [])]


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 found something, 1 found nothing, 2 could not search."""
    if hasattr(sys.stdout, "reconfigure"):
        # Into a pipe or a file, UTF-8, so that file names come through whole for the next program; on a
        # console, its own encoding, with what it cannot show replaced.
        sys.stdout.reconfigure(errors="replace", **({} if sys.stdout.isatty() else {"encoding": "utf-8"}))
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(_split_search(argv))
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "search" and not args.paths:
        args.paths = ["."]
    try:
        return {"search": cmd_search, "index": cmd_index, "status": cmd_status, "forget": cmd_forget, "models": cmd_models}[args.command](args)
    except (EmbedError, OSError, RuntimeError, ValueError, sqlite3.Error) as e:
        _err(str(e))
        return 2
    except Exception as e:  # noqa: BLE001 - never a traceback, and never exit 1, which means "nothing found"
        if os.environ.get("SFIND_DEBUG"):
            raise
        _err(f"unexpected error: {type(e).__name__}: {e} (run with SFIND_DEBUG=1 for details, and please report it)")
        return 2
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
