"""The index: one SQLite file per searched folder, in the data folder (never inside the folder itself).

It holds, for each file, its size and modification time (so that only changed files are read again),
its passages with their line numbers, the keyword terms of each passage in an FTS5 table (BM25 ranking),
and each passage's vector as float16. Searching ranks passages both ways and fuses the two rankings
with reciprocal-rank fusion (RRF): a passage high in either list comes out high, one high in both
comes out on top.
"""

from __future__ import annotations

import fnmatch
import hashlib
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embed import data_home
from .files import MAX_FILE, discover
from .text import HEADED, join_pairs, passages, query_terms, read_text, terms

SCHEMA = 1
RRF_K = 60  # the usual constant: it damps the difference between rank 1 and rank 2
CANDIDATES = 300  # from each ranking, before fusing
# A passage found only by meaning is kept when it is at least this close to the best one: a ranking by
# similarity always has a "best", so without a floor every search would fill its list with noise.
RELATIVE_FLOOR = 0.5
MIN_SIMILARITY = 0.1  # and never below this, however weak the best one is


def index_path(root: Path) -> Path:
    key = hashlib.sha256(str(root.resolve()).lower().encode("utf-8")).hexdigest()[:20]
    return data_home() / "indexes" / f"{key}.sqlite"


@dataclass
class Stats:
    files: int = 0
    read: int = 0  # new or changed files read this time
    removed: int = 0
    passages: int = 0
    embedded: int = 0
    skipped: int = 0  # no text in them (binary, too big, unreadable)
    seconds: float = 0.0


@dataclass
class Hit:
    path: str  # relative to the root, with forward slashes
    line: int
    page: int | None
    text: str
    score: float
    words: list[str] = field(default_factory=list)  # query terms found in the passage
    similarity: float | None = None  # cosine similarity of meaning, when there are vectors
    more: int = 0  # other matching passages in the same file
    context: str = ""  # the heading the passage is under


class Index:
    def __init__(self, root: Path, path: Path | None = None):
        self.root = root.resolve()
        self.path = path or index_path(self.root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._create()
        self._vectors: tuple[np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        self.db.close()

    def _create(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, size INTEGER, mtime_ns INTEGER);
            CREATE TABLE IF NOT EXISTS passages (id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                line INTEGER, page INTEGER, text TEXT, context TEXT, vec BLOB);
            CREATE INDEX IF NOT EXISTS passages_file ON passages(file_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(body, path, tokenize='unicode61 remove_diacritics 2');
        """)
        if self.meta("schema") is None:
            self.set_meta("schema", str(SCHEMA))
            self.set_meta("root", str(self.root))
        elif int(self.meta("schema")) != SCHEMA:
            raise RuntimeError(f"the index {self.path} is from another version of semantic-find; delete it with `sfind forget`")
        self.db.commit()

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    # ---------------------------------------------------------------- updating

    def update(self, model=None, model_name: str | None = None, progress: Callable[[str], None] | None = None) -> Stats:
        """Bring the index up to date with the folder: read new and changed files, drop deleted ones, and
        give every passage a vector from `model` (all of them again if the model changed)."""
        t0 = time.perf_counter()
        st = Stats()
        if model is not None and self.meta("model") != model_name:
            self.db.execute("UPDATE passages SET vec=NULL")
            self.set_meta("model", model_name or "")
            self.set_meta("dim", str(model.dim))
        known = {p: (i, s, m) for i, p, s, m in self.db.execute("SELECT id, path, size, mtime_ns FROM files")}
        seen: set[str] = set()
        for path in discover(self.root):
            try:
                info = path.stat()
            except OSError:
                continue
            rel = path.relative_to(self.root).as_posix() if path != self.root else path.name
            if info.st_size > MAX_FILE:
                continue
            seen.add(rel)
            st.files += 1
            old = known.get(rel)
            if old and old[1] == info.st_size and old[2] == info.st_mtime_ns:
                continue
            if old:
                self._remove(old[0], rel)
            try:
                text = read_text(path)
            except OSError:
                text = None
            cur = self.db.execute("INSERT INTO files(path, size, mtime_ns) VALUES (?, ?, ?)", (rel, info.st_size, info.st_mtime_ns))
            st.read += 1
            if not text:
                st.skipped += 1
                continue
            path_terms = " ".join(terms(rel.replace("/", " ")))
            for p in passages(text, headings=path.suffix.lower() in HEADED):
                c = self.db.execute("INSERT INTO passages(file_id, line, page, text, context) VALUES (?, ?, ?, ?, ?)",
                                    (cur.lastrowid, p.line, p.page, p.text, p.context))
                self.db.execute("INSERT INTO fts(rowid, body, path) VALUES (?, ?, ?)",
                                (c.lastrowid, " ".join(terms(p.context + "\n" + p.text)), path_terms))
                st.passages += 1
            if progress and st.read % 200 == 0:
                progress(f"read {st.read} files")
        for rel, (fid, _, _) in known.items():
            if rel not in seen:
                self._remove(fid, rel)
                st.removed += 1
        self.db.commit()
        if model is not None:
            st.embedded = self._embed_missing(model, progress)
        self.db.commit()
        self._vectors = None
        st.seconds = time.perf_counter() - t0
        return st

    def _remove(self, file_id: int, rel: str) -> None:
        self.db.execute("DELETE FROM fts WHERE rowid IN (SELECT id FROM passages WHERE file_id=?)", (file_id,))
        self.db.execute("DELETE FROM files WHERE id=?", (file_id,))

    def _embed_missing(self, model, progress=None, batch: int = 2000) -> int:
        done = 0
        while True:
            rows = self.db.execute("SELECT p.id, p.text, p.context, f.path FROM passages p JOIN files f ON f.id = p.file_id "
                                   "WHERE p.vec IS NULL LIMIT ?", (batch,)).fetchall()
            if not rows:
                return done
            # The file's name and the heading say what a passage is about as much as its words do.
            texts = [f"{Path(path).stem.replace('_', ' ').replace('-', ' ')}\n{context}\n{text}" for _, text, context, path in rows]
            vecs = model.encode(texts).astype(np.float16)
            self.db.executemany("UPDATE passages SET vec=? WHERE id=?", [(v.tobytes(), r[0]) for v, r in zip(vecs, rows, strict=True)])
            self.db.commit()
            done += len(rows)
            if progress:
                progress(f"embedded {done} passages")

    # ---------------------------------------------------------------- searching

    def counts(self) -> dict[str, int]:
        files = self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n, with_vec = self.db.execute("SELECT COUNT(*), COUNT(vec) FROM passages").fetchone()
        return {"files": files, "passages": n, "embedded": with_vec}

    def _load_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None:
            rows = self.db.execute("SELECT id, vec FROM passages WHERE vec IS NOT NULL").fetchall()
            dim = int(self.meta("dim") or 0)
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), dim) if rows else np.zeros((0, dim), np.float16)
            self._vectors = (ids, mat)
        return self._vectors

    def keyword(self, query: str, limit: int = CANDIDATES) -> list[int]:
        q = query_terms(query)
        if not q:
            return []
        expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in q)
        rows = self.db.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 1.0, 0.4) LIMIT ?", (expr, limit)).fetchall()
        return [r[0] for r in rows]

    def semantic(self, vector: np.ndarray, limit: int = CANDIDATES) -> list[tuple[int, float]]:
        ids, mat = self._load_vectors()
        if not len(ids):
            return []
        sims = mat.astype(np.float32) @ vector.astype(np.float32)
        top = np.argsort(-sims)[:limit]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def search(self, query: str, model=None, *, k: int = 10, mode: str = "hybrid", glob: str | None = None,
               per_file: bool = True) -> list[Hit]:
        """The best `k` passages for `query` (by default the best one per file). `mode`: "hybrid" (both
        rankings fused), "keyword" or "meaning". A model is needed for meaning; without vectors in the
        index, hybrid is keyword search."""
        kw = self.keyword(query) if mode in ("hybrid", "keyword") else []
        sem: list[tuple[int, float]] = []
        if mode in ("hybrid", "meaning") and model is not None and self.counts()["embedded"]:
            sem = self.semantic(model.encode([query])[0])
        if mode == "meaning" and model is None:
            raise RuntimeError("searching by meaning needs a model")
        score: dict[int, float] = {}
        for rank, pid in enumerate(kw):
            score[pid] = score.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        similarity = dict(sem)
        floor = max(MIN_SIMILARITY, sem[0][1] * RELATIVE_FLOOR) if sem else MIN_SIMILARITY
        found_by_words = set(kw)
        for rank, (pid, sim) in enumerate(sem):
            if sim < floor and pid not in found_by_words:
                continue
            score[pid] = score.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if not score:
            return []
        order = sorted(score, key=score.get, reverse=True)
        rows = {}
        for start in range(0, len(order), 500):
            part = order[start:start + 500]
            marks = ",".join("?" * len(part))
            for pid, path, line, page, text, context in self.db.execute(
                    f"SELECT p.id, f.path, p.line, p.page, p.text, p.context FROM passages p JOIN files f ON f.id=p.file_id WHERE p.id IN ({marks})", part):
                rows[pid] = (path, line, page, text, context)
        q_terms = set(query_terms(query))
        hits: list[Hit] = []
        by_file: dict[str, Hit] = {}
        for pid in order:
            if pid not in rows:
                continue
            path, line, page, text, context = rows[pid]
            if glob and not (fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(path.rsplit("/", 1)[-1], glob)):
                continue
            if per_file and path in by_file:
                by_file[path].more += 1
                continue
            found = join_pairs(query, [t for t in dict.fromkeys(terms(text)) if t in q_terms])
            hit = Hit(path, line, page, text, score[pid], found, similarity.get(pid), context=context or "")
            hits.append(hit)
            by_file[path] = hit
            if len(hits) >= k and not per_file:
                break
        return hits[:k]
