"""Test the tests: plant one plausible bug at a time in a copy of the code and check that a test fails.

    python scripts/mutate.py            # all mutants
    python scripts/mutate.py 3 17       # only these (numbers as printed)

The unmutated suite must pass first; otherwise every mutant would look "caught". The repository is
never modified: the code is copied to a temporary folder. Each original must appear exactly once in
its file, so that a mutant never lands somewhere it was not meant for.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IX, TX, CL, EM, FI = (f"src/semantic_find/{m}.py" for m in ("index", "text", "cli", "embed", "files"))

MUTANTS = [
    # what a search shows
    (IX, "full = {pid for pid in kw if cover.get(pid, 0.0) >= FULL_COVERAGE}", "full = set()"),
    (IX, "if s >= SCORE_FLOOR * top or pid in full}", "if s >= SCORE_FLOOR * top}"),
    (IX, "if s >= SCORE_FLOOR * top or pid in full}", "if True}"),
    (IX, "                    if sim < standout:\n                        weak.add(pid)", "                    if False:\n                        weak.add(pid)"),
    (IX, "                if c < 0.5 and pid in similarity and similarity[pid] < standout:", "                if False:"),
    (IX, "            if any(pid not in weak for pid in score):", "            if False:"),
    (IX, "        if hits and all(h.weak for h in hits):\n            return hits[:min(k, WEAK_SHOWN)]", "        if False:\n            return hits"),
    (IX, "if self.last_count >= STANDOUT_FROM else -1.0", "if True else -1.0"),
    (IX, "floor = max(MIN_SIMILARITY, sem[0][1] * RELATIVE_FLOOR) if sem else MIN_SIMILARITY", "floor = MIN_SIMILARITY"),
    (IX, "meaning = max(similarity[pid], 0.0) / best if pid in similarity else c", "meaning = max(similarity.get(pid, 0.0), 0.0) / best"),
    (IX, "score[pid] = KEYWORD_WEIGHT * c * (1 - rank / (10 * len(kw))) + meaning", "score[pid] = meaning"),
    (IX, "            best = max(best if best is not None else sem[0][1], 1e-9)", "            best = max(sem[0][1], 1e-9)"),
    (IX, "weight = {t: math.log(1 + n / df.get(t, 0.5)) for t in q}", "weight = {t: math.log(1 + n / df[t]) for t in q if df.get(t)}"),
    (IX, "weight = {t: math.log(1 + n / df.get(t, 0.5)) for t in q}", "weight = dict.fromkeys(q, 1.0)"),
    (IX, "        self.last_spread = (float(sims.mean()), float(sims.std()))\n        self.last_count = len(sims)\n        if files is not None:",
     "        if files is not None:"),
    # what is grouped and collapsed
    (IX, "            if per_file and path in by_file:\n                by_file[path].more += 1\n                continue", "            pass"),
    (IX, "            if same_text in by_text:", "            if False:"),
    (IX, "                if path not in by_text[same_text].same and path != by_text[same_text].path:", "                if True:"),
    (IX, "            by_text[same_text] = hit", "            pass"),
    # keeping the index right
    (FI, "            lists.append([n for n in dict.fromkeys(names) if n])", "            lists.append([n for n in names if n])"),
    (IX, "            if old and old[1] == info.st_size and old[2] == info.st_mtime_ns:", "            if old:"),
    (IX, "            except Exception:  # noqa: BLE001 - one unreadable file", "            except OSError:  # noqa: BLE001 - one unreadable file"),
    (IX, '            self.db.execute("UPDATE passages SET vec=NULL")', "            pass"),
    (IX, '            if stamp.read_text(encoding="ascii") == version:', "            if True:"),
    (IX, '            self.set_meta("vectors_version", str(int(self.meta("vectors_version") or 0) + 1))', "            pass"),
    (IX, '    if sys.platform in ("win32", "darwin"):  # file names there ignore case', "    if True:  # file names there ignore case"),
    (IX, "            keep = np.isin(file_ids, np.fromiter(files, dtype=np.int64, count=len(files)))", "            keep = np.ones(len(ids), bool)"),
    # terms
    (TX, "    return word.lower()\n    return", "    return word\n    return"),
    (TX, 'return "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))', "return word.lower()"),
    (TX, "        return tuple(tok[i:i + 2] for i in range(len(tok) - 1)) if len(tok) > 1 else (tok,)", "        return (tok,)"),
    (TX, '    out.extend(c for run in re.findall(rf"[{CJK}]{{2,}}", text) for c in run)', "    pass"),
    (TX, "            out.extend(c for c in t if c not in STOP_CHARS)", "            out.append(t)"),
    (TX, "    kept = [t for t in all_terms if t not in STOP]", "    kept = all_terms"),
    (TX, "        if enc == \"euc_kr\" and wide and", "        if False and"),
    (TX, "                raise ValueError(\"too large unpacked\")", "                pass"),
    (TX, "        h = HEADING.match(line) if headings else None", "        h = HEADING.match(line)"),
    # the command line
    (CL, "    keep = [(r, p) for r, p in resolved if not any(o != r and o.is_dir() and r.is_relative_to(o) for o, _ in resolved)]",
     "    keep = resolved"),
    (CL, "        results = [(r, h) for r, h in results if h.score >= SCORE_FLOOR * top or h.full]", "        pass"),
    (CL, "    found = bool(results) and not all(h.weak for _, h in results)", "    found = bool(results)"),
    (CL, "    while n_paths < len(plain) - 1 and _looks_like_path(plain[len(plain) - 1 - n_paths]):", "    while n_paths < len(plain) - 1 and False:"),
    (CL, '        sys.stdout.reconfigure(errors="replace", **({} if sys.stdout.isatty() else {"encoding": "utf-8"}))',
     '        sys.stdout.reconfigure(errors="replace")'),
    (CL, "    if args.top < 1:", "    if False:"),
    # models and downloads
    (EM, "        if _sha256(part) != sha:", "        if False:"),
    (EM, "        if have > size:\n            part.unlink()", "        if False:\n            part.unlink()"),
    (EM, "                ids = [t for t in enc.ids if t != self.unk]", "                ids = list(enc.ids)"),
    (EM, '        return ipaddress.ip_address(host).is_loopback or host in ("0.0.0.0", "::")', "        return True"),
    (FI, "        for rel in untracked:  # an untracked node_modules or .venv is clutter, ignored or not\n"
        "            if not _ignored(rel, patterns) and not any(part in SKIP_DIRS for part in rel.split(\"/\")[:-1]):",
     "        for rel in untracked:  # an untracked node_modules or .venv is clutter, ignored or not\n            if not _ignored(rel, patterns):"),
]


def main(argv: list[str]) -> int:
    chosen = {int(a) for a in argv} if argv else None
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"))
        env = {**os.environ, "PYTHONPATH": str(copy / "src") + os.pathsep + str(copy / "tests"), "PYTHONDONTWRITEBYTECODE": "1"}

        def run_suite() -> subprocess.CompletedProcess:
            return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
                                  cwd=copy, env=env, capture_output=True, timeout=300)

        check = subprocess.run([sys.executable, "-c", "import semantic_find; print(semantic_find.__file__)"], cwd=copy, env=env,
                               capture_output=True, text=True)
        if not check.stdout.strip().startswith(str(copy)):
            print(f"semantic_find is imported from {check.stdout.strip()}, not from the copy")
            return 2
        baseline = run_suite()
        if baseline.returncode != 0:
            print("The test suite fails without any mutant; fix that first:\n" + baseline.stdout.decode(errors="replace")[-2000:])
            return 2
        survived = []
        for i, (file, original, mutant) in enumerate(MUTANTS, start=1):
            if chosen and i not in chosen:
                continue
            path = copy / file
            source = path.read_text(encoding="utf-8")
            if source.count(original) != 1:
                print(f"{i:>2}. FOUND {source.count(original)} TIMES in {file}: {original[:60]} (update this list)")
                survived.append(i)
                continue
            path.write_text(source.replace(original, mutant, 1), encoding="utf-8")
            try:
                caught = run_suite().returncode != 0
            except subprocess.TimeoutExpired:
                caught = True  # a hang is a failure too
            path.write_text(source, encoding="utf-8")
            print(f"{i:>2}. {'caught  ' if caught else 'SURVIVED'}  {file}: {original[:70]}", flush=True)
            if not caught:
                survived.append(i)
        total = len(chosen) if chosen else len(MUTANTS)
        print(f"\n{total - len(survived)} of {total} mutants caught" + (f"; survived: {' '.join(map(str, survived))}" if survived else ""))
        return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
