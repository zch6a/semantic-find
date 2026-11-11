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
import sys
from pathlib import Path

from . import __version__
from .embed import DEFAULT_MODEL, MODELS, EmbedError, download, is_downloaded, load, model_dir
from .index import Hit, Index, index_path

COMMANDS = {"index", "status", "forget", "models", "search"}


def _err(msg: str) -> None:
    print(f"sfind: {msg}", file=sys.stderr)


def _model_name(args, idx: Index | None = None) -> str:
    return args.model or os.environ.get("SFIND_MODEL") or (idx.meta("model") if idx else None) or DEFAULT_MODEL


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _progress_bar(file: str, done: int, total: int) -> None:
    if total and sys.stderr.isatty():
        print(f"\r  {file}: {done / 1048576:6.1f} of {total / 1048576:.1f} MB", end="", file=sys.stderr, flush=True)


def get_model(name: str, *, ask: bool):
    """The model, or None when it is a built-in one not downloaded yet and the user did not agree to
    download it now (searching then matches keywords only)."""
    if name in MODELS and not is_downloaded(name):
        info = MODELS[name]
        if not ask or not _interactive():
            _err(f"searching by keywords only: the model {name} is not downloaded. `sfind models download` fetches it "
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

    st = idx.update(model, name if model is not None else None, progress=progress if _interactive() else None)
    if not quiet and (st.read or st.removed or st.embedded):
        print("\r", end="", file=sys.stderr)
        parts = [f"{st.read} file{'s' if st.read != 1 else ''} read"]
        if st.removed:
            parts.append(f"{st.removed} gone")
        if st.embedded:
            parts.append(f"{st.embedded} passages embedded")
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
    out = flat[start:start + width]
    return ("…" if start else "") + out + ("…" if start + width < len(flat) else "")


def highlight(text: str, words: list[str]) -> str:
    if not words or not _color():
        return text
    pattern = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)), re.I)
    return pattern.sub(lambda m: f"\033[1;33m{m.group(0)}\033[0m", text)


def _where(root: Path, hit: Hit) -> str:
    full = root / hit.path if root.is_dir() else root
    try:
        shown = os.path.relpath(full)
        shown = shown if not shown.startswith("..") or len(shown) < len(str(full)) else str(full)
    except ValueError:  # another drive on Windows
        shown = str(full)
    return f"{shown} (page {hit.page})" if hit.page else f"{shown}:{hit.line}"


def why(hit: Hit) -> str:
    parts = []
    if hit.words:
        parts.append("words: " + ", ".join(hit.words[:6]))
    if hit.similarity is not None:
        parts.append(f"meaning {hit.similarity:.2f}")
    if hit.more:
        parts.append(f"{hit.more} more in this file")
    return " · ".join(parts)


# ---------------------------------------------------------------- commands

def cmd_search(args) -> int:
    query = " ".join(args.query).strip()
    if not query:
        _err("what should be searched for?")
        return 2
    if args.mode == "keyword":
        model = name = None
    else:
        first = _open(args.paths[0])
        name = _model_name(args, first)
        first.close()
        model = get_model(name, ask=True)
        if model is None and args.mode == "meaning":
            return 2
    results: list[tuple[Path, Hit]] = []
    for path in args.paths:
        idx = _open(path)
        try:
            if not args.no_update:
                _update(idx, model, name, args.quiet)
            for hit in idx.search(query, model, k=args.top, mode=args.mode if model is not None else "keyword", glob=args.glob,
                                  per_file=not args.all):
                results.append((idx.root, hit))
        finally:
            idx.close()
    results.sort(key=lambda r: r[1].score, reverse=True)
    results = results[:args.top]
    if args.json:
        print(json.dumps([{"path": str(root / h.path if root.is_dir() else root), "line": h.line, "page": h.page, "text": h.text,
                           "score": round(h.score, 6), "words": h.words,
                           "similarity": None if h.similarity is None else round(h.similarity, 4), "more": h.more}
                          for root, h in results], ensure_ascii=False, indent=2))
        return 0 if results else 1
    if args.files:
        for root, h in dict(((r / h.path).as_posix(), (r, h)) for r, h in results).values():
            print(str(root / h.path) if root.is_dir() else str(root))
        return 0 if results else 1
    if not results:
        _err("nothing found")
        return 1
    color = _color()
    for root, h in results:
        where = _where(root, h)
        print(f"\033[1;36m{where}\033[0m" if color else where)
        words = sorted(h.words, key=len, reverse=True)
        print("    " + highlight(snippet(h.text, words, heading=h.context), words))
        detail = why(h)
        if detail:
            print(f"    \033[2m({detail})\033[0m" if color else f"    ({detail})")
    return 0


def cmd_index(args) -> int:
    for path in args.paths:
        idx = _open(path)
        try:
            name = _model_name(args, idx)
            model = None if args.keyword_only else get_model(name, ask=True)
            _update(idx, model, name, quiet=False)
            c = idx.counts()
            _err(f"{idx.root}: {c['files']} files, {c['passages']} passages, {c['embedded']} with vectors")
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
            print(f"{p}\n  {c['files']} files, {c['passages']} passages, {c['embedded']} with vectors ({idx.meta('model') or 'no model'})\n"
                  f"  index: {db} ({size:.1f} MB)")
        finally:
            idx.close()
    return 0


def cmd_forget(args) -> int:
    for path in args.paths:
        p = Path(path).expanduser().resolve()
        db = index_path(p)
        gone = [f for f in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")) if f.exists()]
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
    """`sfind QUERY [PATH...]`: the words before the first argument that is a path are the query, the
    rest are paths. `sfind search ... --in PATH` is the explicit form."""
    if not argv or argv[0] in COMMANDS or argv[0] in ("-h", "--help", "--version"):
        return argv
    words, paths, opts = [], [], []
    it = iter(range(len(argv)))
    takes_value = {"-k", "--top", "-g", "--glob", "--mode", "--model", "--in"}
    for i in it:
        a = argv[i]
        if a.startswith("-") and len(a) > 1:
            opts.append(a)
            if a in takes_value and i + 1 < len(argv):
                opts.append(argv[i + 1])
                next(it, None)
        elif words and _looks_like_path(a):
            paths.append(a)
        elif paths:
            paths.append(a)  # (a path that does not exist: reported when it is opened)
        else:
            words.append(a)
    return ["search", *words, *opts, *(["--in", *paths] if paths else [])]


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 found something, 1 found nothing, 2 could not search."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
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
    except (EmbedError, OSError, RuntimeError, ValueError) as e:
        _err(str(e))
        return 2
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
