"""Which files to search. In a git repository, what git tracks or would track (so .gitignore is
honoured exactly as git reads it); elsewhere, every file except well-known clutter: version-control
folders, dependency and build folders, caches, and hidden folders."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", ".tox", ".idea", ".vscode", "dist", "build", "target", ".next", ".nuxt", ".cache",
             "site-packages", "$RECYCLE.BIN", "System Volume Information"}
MAX_FILE = 50 * 1024 * 1024  # documents larger than this are skipped
IGNORE_FILE = ".sfindignore"  # extra patterns, one per line, as in .gitignore (a simplified reading)


def _git_files(root: Path) -> list[Path] | None:
    git = shutil.which("git")
    if not git:
        return None
    try:
        inside = subprocess.run([git, "-C", str(root), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, timeout=10)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        out = subprocess.run([git, "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    names = out.stdout.decode("utf-8", "surrogateescape").split("\0")
    return [root / n for n in names if n]


def _patterns(root: Path) -> list[str]:
    p = root / IGNORE_FILE
    if not p.is_file():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return [x.strip().rstrip("/") for x in lines if x.strip() and not x.lstrip().startswith("#")]


def _ignored(rel: str, patterns: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for pat in patterns:
        anchored = pat.startswith("/")
        pat = pat.lstrip("/")
        if fnmatch.fnmatch(rel, pat) or (not anchored and "/" not in pat and fnmatch.fnmatch(name, pat)):
            return True
        if fnmatch.fnmatch(rel, pat + "/*") or any(fnmatch.fnmatch(part, pat) for part in rel.split("/")[:-1] if not anchored and "/" not in pat):
            return True
    return False


def discover(root: Path) -> Iterator[Path]:
    """Files under `root` worth searching, as absolute paths. A single file is itself."""
    root = root.resolve()
    if root.is_file():
        yield root
        return
    patterns = _patterns(root)
    listed = _git_files(root)
    if listed is not None:
        for p in listed:
            rel = p.relative_to(root).as_posix()
            if not _ignored(rel, patterns) and not any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
                yield p
        return
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        rel_dir = base.relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") and not _ignored(rel_dir + d, patterns))
        for f in sorted(filenames):
            if f.startswith(".") or _ignored(rel_dir + f, patterns):
                continue
            yield base / f
