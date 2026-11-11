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
from .text import HEADED, coverage_terms, index_terms, join_pairs, passages, query_terms, read_text, terms

SCHEMA = 3  # 2: terms folded and split as FTS5 stores them; 3: FTS5 keeps no second copy of the terms
# SQLite 3.43 (2023) can delete rows from a contentless FTS5 table: then the index keeps only its
# inverted lists, not a copy of every passage's terms (142 MB of 337 for 77,000 passages). Older SQLite
# gets the table with its copy.
CONTENTLESS = sqlite3.sqlite_version_info >= (3, 43, 0)
FTS_OPTIONS = ", content='', contentless_delete=1" if CONTENTLESS else ""
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
FULL_COVERAGE = 0.99  # a passage with (nearly) every word of the query is kept whatever its score
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
    full: bool = False  # contains every word of the query: never left out


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
        old = self.db.execute("SELECT name FROM sqlite_master WHERE name='meta'").fetchone()
        if old and (self.meta("schema") or "0") != str(SCHEMA):
            # An index from another version is a cache: it is built again, not an error.
            self.db.executescript("DROP TABLE IF EXISTS fts_vocab; DROP TABLE IF EXISTS fts; DROP TABLE IF EXISTS passages; "
                                  "DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS meta;")
            old = None
        if not old:
            # Space freed by deleted files goes back to the disk (incremental_vacuum after an update);
            # the setting takes effect on an empty database, or with the VACUUM that follows.
            self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
            self.db.execute("VACUUM")
        self.db.executescript(f"""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, size INTEGER, mtime_ns INTEGER);
            CREATE TABLE IF NOT EXISTS passages (id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                line INTEGER, page INTEGER, text TEXT, context TEXT, vec BLOB);
            CREATE INDEX IF NOT EXISTS passages_file ON passages(file_id);
            CREATE INDEX IF NOT EXISTS passages_novec ON passages(id) WHERE vec IS NULL;
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(body, path, tokenize='unicode61 remove_diacritics 0'{FTS_OPTIONS});
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_vocab USING fts5vocab(fts, 'row');
        """)
        if self.meta("schema") is None:
            self.set_meta("schema", str(SCHEMA))
            self.set_meta("root", str(self.root))
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
        base = str(self.root).rstrip("\\/") + os.sep
        for path in discover(self.root):
            full = str(path)
            try:
                info = os.stat(full)
            except OSError:
                continue
            # (string slicing, not pathlib: relative_to took 0.6 s for 6,000 files on every search)
            rel = full[len(base):].replace(os.sep, "/") if full.startswith(base) else path.name
            if info.st_size > MAX_FILE:
                continue
            seen.add(rel)  # (files.discover lists each file once, a merge conflict's included)
            st.files += 1
            old = known.get(rel)
            if old and old[1] == info.st_size and old[2] == info.st_mtime_ns:
                continue
            try:
                rel.encode("utf-8")
            except UnicodeEncodeError:  # a file name that is not valid Unicode (bytes from another system)
                st.skipped += 1
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
            path_terms = " ".join(index_terms(rel.replace("/", " ")))
            # All of a file's passages in two statements, with their ids given here (one INSERT per passage
            # took 9 of 65 s for 77,000 passages).
            ps = passages(text, headings=path.suffix.lower() in HEADED)
            first = (self.db.execute("SELECT COALESCE(MAX(id), 0) FROM passages").fetchone()[0] or 0) + 1
            self.db.executemany("INSERT INTO passages(id, file_id, line, page, text, context) VALUES (?, ?, ?, ?, ?, ?)",
                                [(first + i, cur.lastrowid, p.line, p.page, p.text, p.context) for i, p in enumerate(ps)])
            self.db.executemany("INSERT INTO fts(rowid, body, path) VALUES (?, ?, ?)",
                                [(first + i, " ".join(index_terms(p.context + "\n" + p.text)), path_terms) for i, p in enumerate(ps)])
            st.passages += len(ps)
            if st.read % COMMIT_EVERY == 0:
                self.db.commit()
                if progress:
                    progress(f"read {st.read} files")
        for rel, (fid, _, _) in known.items():
            if rel not in seen:
                self._remove(fid, rel)
                st.removed += 1
        self.db.commit()
        if st.removed or st.read:
            self.db.execute("PRAGMA incremental_vacuum")  # give back the pages of what was removed or replaced
        if model is not None:
            st.embedded = self._embed_missing(model, progress)
        if st.read or st.removed or st.embedded:
            self.set_meta("vectors_version", str(int(self.meta("vectors_version") or 0) + 1))
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
        n = self.db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        missing = self.db.execute("SELECT COUNT(*) FROM passages WHERE vec IS NULL").fetchone()[0]  # (a partial index)
        return {"files": files, "passages": n, "embedded": n - missing}

    def _load_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All vectors as one matrix, with each one's passage and file. Kept in a .npy file next to the
        index and memory-mapped, rewritten only when the vectors change: rebuilding the matrix from the
        rows took 0.4 s per search at 77,000 passages."""
        if self._vectors is not None:
            return self._vectors
        version = self.meta("vectors_version") or "0"
        cache = self.path.with_name(self.path.stem + ".vectors.npy")
        keys = self.path.with_name(self.path.stem + ".vectors-ids.npy")
        stamp = self.path.with_name(self.path.stem + ".vectors-version")
        try:
            if stamp.read_text(encoding="ascii") == version:
                ids_files = np.load(keys)
                self._vectors = (ids_files[0], ids_files[1], np.load(cache, mmap_mode="r"))
                return self._vectors
        except (OSError, ValueError):
            pass
        rows = self.db.execute("SELECT id, file_id, vec FROM passages WHERE vec IS NOT NULL").fetchall()
        dim = int(self.meta("dim") or 0)
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        files = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
        mat = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float16).reshape(len(rows), dim) if rows else np.zeros((0, dim), np.float16)
        self._vectors = (ids, files, mat)
        with contextlib.suppress(OSError):  # a cache: if it cannot be written, the next search builds it again
            stamp.unlink(missing_ok=True)
            np.save(cache, mat)
            np.save(keys, np.vstack([ids, files]))
            _private(cache)
            _private(keys)
            stamp.write_text(version, encoding="ascii")
        return self._vectors

    def _allowed_files(self, glob: str | None) -> set[int] | None:
        """The files a -g pattern lets through (None: all). Applied before ranking, so that the
        matches in those files are not crowded out of the candidates by others."""
        if not glob:
            return None
        return {fid for fid, path in self.db.execute("SELECT id, path FROM files")
                if fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(path.rsplit("/", 1)[-1], glob)}

    def keyword(self, query: str, limit: int | None = None, files: set[int] | None = None) -> list[int]:
        limit = limit or CANDIDATES  # (read now, not when the function was defined)
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
        q = coverage_terms(query)
        if not q or not pids:
            return {}
        marks = ",".join("?" * len(q))
        df = dict(self.db.execute(f"SELECT term, doc FROM fts_vocab WHERE term IN ({marks})", q).fetchall())
        n = self.db.execute("SELECT COUNT(*) FROM fts").fetchone()[0] or 1
        # A word in no passage at all weighs the most: "xylophone lessons" is not covered by "lessons".
        weight = {t: math.log(1 + n / df.get(t, 0.5)) for t in q}
        total = sum(weight.values())
        if not total:
            return {}
        out = {}
        for start in range(0, len(pids), 500):
            part = pids[start:start + 500]
            # The passage's terms, worked out again from its text (FTS5 keeps no copy of them).
            for pid, text, context, path in self.db.execute(
                    f"SELECT p.id, p.text, p.context, f.path FROM passages p JOIN files f ON f.id = p.file_id "
                    f"WHERE p.id IN ({','.join('?' * len(part))})", part):
                have = set(index_terms((context or "") + "\n" + text)) | set(index_terms(path.replace("/", " ")))
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

    def semantic(self, vector: np.ndarray, limit: int | None = None, files: set[int] | None = None) -> list[tuple[int, float]]:
        limit = limit or CANDIDATES
        ids, file_ids, mat = self._load_vectors()
        if not len(ids):
            return []
        if vector.shape[0] != mat.shape[1]:
            raise RuntimeError(f"the index was built with {self.meta('model')}, whose vectors do not fit this model's; "
                               "search without --no-update to embed it again, or pass --model " + str(self.meta("model")))
        sims = mat.astype(np.float32) @ vector.astype(np.float32)
        # The background for the stand-out test is the whole folder, whatever -g lets through.
        self.last_spread = (float(sims.mean()), float(sims.std()))
        self.last_count = len(sims)
        if files is not None:
            keep = np.isin(file_ids, np.fromiter(files, dtype=np.int64, count=len(files)))
            ids, sims = ids[keep], sims[keep]
        top = np.argsort(-sims)[:limit]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def best_similarity(self, vector: np.ndarray) -> float:
        """The highest similarity of any passage to the query: searches of several folders are scored
        against the best of all of them, so that their scores can be compared."""
        _, _, mat = self._load_vectors()
        return float((mat.astype(np.float32) @ vector.astype(np.float32)).max()) if len(mat) else 0.0

    def search(self, query: str, model=None, *, k: int = 10, mode: str = "hybrid", glob: str | None = None,
               per_file: bool = True, query_vector: np.ndarray | None = None, best: float | None = None) -> list[Hit]:
        """The best `k` passages for `query` (by default the best one per file). `mode`: "hybrid" (both
        rankings fused), "keyword" or "meaning". A model is needed for meaning; without vectors in the
        index, hybrid is keyword search. `best`: the similarity to score against (the best over several
        folders searched together; by default this folder's best)."""
        allowed = self._allowed_files(glob)
        kw = self.keyword(query, files=allowed) if mode in ("hybrid", "keyword") else []
        sem: list[tuple[int, float]] = []
        if mode == "meaning" and model is None:
            raise RuntimeError("searching by meaning needs a model")
        if mode in ("hybrid", "meaning") and model is not None:
            if not self.counts()["embedded"]:
                if mode == "meaning":
                    raise RuntimeError(f"the index of {self.root} has no vectors yet: search once without --no-update to embed it")
            else:
                if query_vector is None:
                    query_vector = model.encode([query])[0]
                sem = self.semantic(query_vector, files=allowed)
        score: dict[int, float] = {}
        weak: set[int] = set()
        similarity = dict(sem)
        floor = max(MIN_SIMILARITY, sem[0][1] * RELATIVE_FLOOR) if sem else MIN_SIMILARITY
        cover: dict[int, float] = {}
        full: set[int] = set()
        if sem:
            best = max(best if best is not None else sem[0][1], 1e-9)
            cover = self.coverage(query, kw)
            full = {pid for pid in kw if cover.get(pid, 0.0) >= FULL_COVERAGE}
            if kw:
                similarity.update(self.similarities(query_vector, [p for p in kw if p not in similarity]))
            mean, spread = self.last_spread
            standout = mean + MIN_STANDOUT * max(spread, 1e-6) if self.last_count >= STANDOUT_FROM else -1.0
            for rank, pid in enumerate(kw):  # BM25's order breaks ties between passages that cover as much
                c = cover.get(pid, 0.0)
                # A passage without a vector yet (indexed by keywords only, or while another sfind embeds
                # it) is judged by its words alone: the share of the query it contains stands for its meaning.
                meaning = max(similarity[pid], 0.0) / best if pid in similarity else c
                score[pid] = KEYWORD_WEIGHT * c * (1 - rank / (10 * len(kw))) + meaning
                if c < 0.5 and pid in similarity and similarity[pid] < standout:
                    weak.add(pid)  # one word of several in common, and nothing in its meaning stands out
            for pid, sim in sem:
                if pid not in score and sim >= floor:
                    score[pid] = sim / best
                    if sim < standout:
                        weak.add(pid)
            if any(pid not in weak for pid in score):
                score = {pid: s for pid, s in score.items() if pid not in weak}
            top = max(score.values(), default=0.0)
            # ... but a passage with every word of the query is never left out: it is what grep would find
            score = {pid: s for pid, s in score.items() if s >= SCORE_FLOOR * top or pid in full}
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
                continue  # (the copy's file may still have other passages of its own to show)
            found = join_pairs(query, [t for t in dict.fromkeys(terms(text)) if t in q_terms])
            hit = Hit(path, line, page, text, score[pid], found, similarity.get(pid), context=context or "", weak=pid in weak,
                      full=pid in full)
            hits.append(hit)
            by_file[path] = hit
            by_text[same_text] = hit
            if len(hits) >= k and not per_file:
                break
        if hits and all(h.weak for h in hits):
            return hits[:min(k, WEAK_SHOWN)]
        return hits[:k]
