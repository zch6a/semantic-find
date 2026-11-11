"""The index: one SQLite file per searched folder, in the data folder (never inside the folder itself).

It holds, for each file, its size and modification time (so that only changed files are read again),
its passages with their line numbers, the keyword terms of each passage in an FTS5 table (BM25 ranking),
and each passage's vector as float16.

Searching scores each candidate passage as

    similarity / best similarity  +  KEYWORD_WEIGHT (0.15) × the share of the query's words it contains

(words weighted by rarity). Reciprocal-rank fusion, the usual way to merge two rankings, was tried first
and measured on eval/: it lets a passage that shares one word of four with the query outrank the one
that means what was asked, and with the default model it found the right file first for only 22% of
queries across languages. Score fusion was then tuned on 100 queries (see KEYWORD_WEIGHT).
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import math
import os
import sqlite3
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embed import data_home
from .files import MAX_FILE, discover
from .text import HEADED, index_terms, join_pairs, passages, query_terms, read_text, terms

SCHEMA = 1
CANDIDATES = 300  # from each ranking, before fusing
# A passage found only by meaning is kept when it is at least this close to the best one: a ranking by
# similarity always has a "best", so without a floor every search would fill its list with noise.
RELATIVE_FLOOR = 0.5
MIN_SIMILARITY = 0.1  # and never below this, however weak the best one is
# A match by meaning alone is "strong" when it stands out from the folder: at least this many standard
# deviations above the mean similarity of all passages to the query. When there are strong results,
# the weak ones are not shown; when there are none, the closest few are shown as weak, and the search
# counts as "nothing found". Weak matches are labelled rather than dropped because no statistic tells
# them apart reliably: measured with eval/standout.py on 4,000 and 64,000 passages, at the threshold that
# keeps 90% of right answers (global z 1.7, the rank-20-300 tail z, the gap to the next 20, or raw
# similarity), at best 30% of searches for subjects absent from the folder were caught; the weakest
# tenth of right answers look like the best of the noise. Applies from STANDOUT_FROM passages: a small
# folder has no background to stand out from.
MIN_STANDOUT = 3.0
STANDOUT_FROM = 500
WEAK_SHOWN = 3  # when nothing is strong: how many of the closest are shown
# How much a passage containing all of the query's words gains over one that only means the same.
# Chosen on eval/ (100 queries, three models, 0 to 1.5 tried): higher loses queries across languages,
# lower loses exact names ("bcrypt", "TokenBucket") with the default model.
KEYWORD_WEIGHT = 0.15
# A result is shown only if it scores at least this share of the best result. When the best one matches
# by both words and meaning, a passage that only means something vaguely similar is noise (measured on
# a 3,000-file folder: "when should the timing belt be checked" listed two unrelated notes after the
# right one). Chosen on eval/.
SCORE_FLOOR = 0.6


COMMIT_EVERY = 200  # files: an interrupted first index keeps what it has done


def index_path(root: Path) -> Path:
    name = str(root.resolve())
    if sys.platform in ("win32", "darwin"):  # file names there ignore case: C:\Notes is c:\notes
        name = name.lower()
    key = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
    return data_home() / "indexes" / f"{key}.sqlite"


class Busy(RuntimeError):
    """Another sfind is updating this index."""


@contextlib.contextmanager
def updating(db_path: Path):
    """One updater at a time per index, held with an OS file lock (released if the process dies)."""
    with open(db_path.with_name(db_path.name + ".lock"), "a+b") as lock:
        try:
            if sys.platform == "win32":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise Busy("another sfind is updating this index") from e
        yield  # closing the file releases the lock


def _private(path: Path) -> None:
    """The index holds the text of your files: readable by you only (on systems with file modes)."""
    if sys.platform != "win32":
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)


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
    same: list[str] = field(default_factory=list)  # other files with exactly the same passage
    weak: bool = False  # found by meaning alone, without standing out: shown only when nothing is strong


class Index:
    def __init__(self, root: Path, path: Path | None = None):
        self.root = root.resolve()
        self.path = path or index_path(self.root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            with contextlib.suppress(OSError):
                os.chmod(self.path.parent, 0o700)
        self.db = sqlite3.connect(self.path, timeout=30)
        _private(self.path)
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
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_vocab USING fts5vocab(fts, 'row');
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
        give every passage a vector from `model` (all of them again if the model changed). Raises Busy
        when another sfind is doing it."""
        with updating(self.path):
            return self._update(model, model_name, progress)

    def _update(self, model, model_name, progress) -> Stats:
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
            if rel in seen:
                continue  # git lists a file in a merge conflict once per side
            seen.add(rel)
            st.files += 1
            old = known.get(rel)
            if old and old[1] == info.st_size and old[2] == info.st_mtime_ns:
                continue
            if old:
                self._remove(old[0], rel)
            try:
                text = read_text(path)
            except Exception:  # noqa: BLE001 - one unreadable file must not stop the rest; it counts as skipped
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
                                (c.lastrowid, " ".join(index_terms(p.context + "\n" + p.text)), path_terms))
                st.passages += 1
            if st.read % COMMIT_EVERY == 0:
                self.db.commit()
                if progress:
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

    def _load_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._vectors is None:
            rows = self.db.execute("SELECT id, file_id, vec FROM passages WHERE vec IS NOT NULL").fetchall()
            dim = int(self.meta("dim") or 0)
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            files = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
            mat = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float16).reshape(len(rows), dim) if rows else np.zeros((0, dim), np.float16)
            self._vectors = (ids, files, mat)
        return self._vectors

    def _allowed_files(self, glob: str | None) -> set[int] | None:
        """The files a -g pattern lets through (None: all). Applied before ranking, so that the
        matches in those files are not crowded out of the candidates by others."""
        if not glob:
            return None
        return {fid for fid, path in self.db.execute("SELECT id, path FROM files")
                if fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(path.rsplit("/", 1)[-1], glob)}

    def keyword(self, query: str, limit: int = CANDIDATES, files: set[int] | None = None) -> list[int]:
        q = query_terms(query)
        if not q or files == set():
            return []
        expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in q)
        if files is None:
            rows = self.db.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 1.0, 0.4) LIMIT ?", (expr, limit)).fetchall()
            return [r[0] for r in rows]
        self.db.execute("CREATE TEMP TABLE IF NOT EXISTS allowed (id INTEGER PRIMARY KEY)")
        self.db.execute("DELETE FROM allowed")
        self.db.executemany("INSERT INTO allowed VALUES (?)", [(f,) for f in files])
        rows = self.db.execute("SELECT rowid FROM fts WHERE fts MATCH ? AND rowid IN "
                               "(SELECT p.id FROM passages p JOIN allowed a ON a.id = p.file_id) ORDER BY bm25(fts, 1.0, 0.4) LIMIT ?",
                               (expr, limit)).fetchall()
        return [r[0] for r in rows]

    def coverage(self, query: str, pids: list[int]) -> dict[int, float]:
        """For each passage, the share of the query's words it contains, each word weighted by its
        rarity (IDF). Words that occur nowhere in the index (such as a character pair that spans two
        Chinese words, "么配" in "怎么配置") are left out of the count."""
        q = query_terms(query)
        if not q or not pids:
            return {}
        marks = ",".join("?" * len(q))
        df = dict(self.db.execute(f"SELECT term, doc FROM fts_vocab WHERE term IN ({marks})", q).fetchall())
        n = self.db.execute("SELECT COUNT(*) FROM fts").fetchone()[0] or 1
        weight = {t: math.log(1 + n / df[t]) for t in q if df.get(t)}
        total = sum(weight.values())
        if not total:
            return {}
        out = {}
        for start in range(0, len(pids), 500):
            part = pids[start:start + 500]
            for pid, body, path in self.db.execute(f"SELECT rowid, body, path FROM fts WHERE rowid IN ({','.join('?' * len(part))})", part):
                have = set(body.split()) | set(path.split())
                out[pid] = sum(w for t, w in weight.items() if t in have) / total
        return out

    def similarities(self, vector: np.ndarray, pids: list[int]) -> dict[int, float]:
        """Similarity of meaning for particular passages (keyword hits beyond the semantic candidates)."""
        ids, _, mat = self._load_vectors()
        if not pids or not len(ids):
            return {}
        where = {int(p): i for i, p in enumerate(ids)}
        rows = [(p, where[p]) for p in pids if p in where]
        if not rows:
            return {}
        sims = mat[[r for _, r in rows]].astype(np.float32) @ vector.astype(np.float32)
        return {p: float(s) for (p, _), s in zip(rows, sims, strict=True)}

    def semantic(self, vector: np.ndarray, limit: int = CANDIDATES, files: set[int] | None = None) -> list[tuple[int, float]]:
        ids, file_ids, mat = self._load_vectors()
        if files is not None:
            keep = np.isin(file_ids, np.fromiter(files, dtype=np.int64, count=len(files)))
            ids, mat = ids[keep], mat[keep]
        if not len(ids):
            return []
        if vector.shape[0] != mat.shape[1]:
            raise RuntimeError(f"the index was built with {self.meta('model')}, whose vectors do not fit this model's; "
                               "search without --no-update to embed it again, or pass --model " + str(self.meta("model")))
        sims = mat.astype(np.float32) @ vector.astype(np.float32)
        top = np.argsort(-sims)[:limit]
        self.last_spread = (float(sims.mean()), float(sims.std()))  # for the stand-out test in search
        self.last_count = len(sims)
        return [(int(ids[i]), float(sims[i])) for i in top]

    def search(self, query: str, model=None, *, k: int = 10, mode: str = "hybrid", glob: str | None = None,
               per_file: bool = True) -> list[Hit]:
        """The best `k` passages for `query` (by default the best one per file). `mode`: "hybrid" (both
        rankings fused), "keyword" or "meaning". A model is needed for meaning; without vectors in the
        index, hybrid is keyword search."""
        allowed = self._allowed_files(glob)
        kw = self.keyword(query, files=allowed) if mode in ("hybrid", "keyword") else []
        sem: list[tuple[int, float]] = []
        query_vector = None
        if mode in ("hybrid", "meaning") and model is not None and self.counts()["embedded"]:
            query_vector = model.encode([query])[0]
            sem = self.semantic(query_vector, files=allowed)
        if mode == "meaning" and model is None:
            raise RuntimeError("searching by meaning needs a model")
        score: dict[int, float] = {}
        weak: set[int] = set()
        similarity = dict(sem)
        floor = max(MIN_SIMILARITY, sem[0][1] * RELATIVE_FLOOR) if sem else MIN_SIMILARITY
        if sem:
            best = max(sem[0][1], 1e-9)
            cover = self.coverage(query, kw)
            if kw:
                similarity.update(self.similarities(query_vector, [p for p in kw if p not in similarity]))
            for rank, pid in enumerate(kw):  # BM25's order breaks ties between passages that cover as much
                score[pid] = KEYWORD_WEIGHT * cover.get(pid, 0.0) * (1 - rank / (10 * len(kw))) + max(similarity.get(pid, 0.0), 0.0) / best
            mean, spread = self.last_spread
            standout = mean + MIN_STANDOUT * max(spread, 1e-6) if self.last_count >= STANDOUT_FROM else -1.0
            for pid, sim in sem:
                if pid not in score and sim >= floor:
                    score[pid] = sim / best
                    if sim < standout:
                        weak.add(pid)
            if any(pid not in weak for pid in score):
                score = {pid: s for pid, s in score.items() if pid not in weak}
            top = max(score.values(), default=0.0)
            score = {pid: s for pid, s in score.items() if s >= SCORE_FLOOR * top}
        else:  # keywords alone: BM25's order
            for rank, pid in enumerate(kw):
                score[pid] = 1.0 / (rank + 1)
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
        by_text: dict[str, Hit] = {}
        for pid in order:
            if pid not in rows:
                continue
            path, line, page, text, context = rows[pid]
            if glob and not (fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(path.rsplit("/", 1)[-1], glob)):
                continue
            if per_file and path in by_file:
                by_file[path].more += 1
                continue
            same_text = " ".join(text.split())
            if same_text in by_text:  # a copy of a passage already listed: said once, with where the copies are
                if path not in by_text[same_text].same and path != by_text[same_text].path:
                    by_text[same_text].same.append(path)
                by_file.setdefault(path, by_text[same_text])
                continue
            found = join_pairs(query, [t for t in dict.fromkeys(terms(text)) if t in q_terms])
            hit = Hit(path, line, page, text, score[pid], found, similarity.get(pid), context=context or "", weak=pid in weak)
            hits.append(hit)
            by_file[path] = hit
            by_text[same_text] = hit
            if len(hits) >= k and not per_file:
                break
        if hits and all(h.weak for h in hits):
            return hits[:min(k, WEAK_SHOWN)]
        return hits[:k]
