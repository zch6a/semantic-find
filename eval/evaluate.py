"""How well each search mode finds the right file, on the evaluation set in eval/data.py.

    python eval/evaluate.py                       # the default model (it must be downloaded)
    python eval/evaluate.py --model potion-base-8M
    python eval/evaluate.py --model ollama:bge-m3
    python eval/evaluate.py --haystack 3000       # the documents hidden among 3,000 generated files

For every query it reports whether the right file is first (hit@1), among the first five (hit@5), and the
mean reciprocal rank (MRR: 1 for first, 1/2 for second, ...), for keyword search, search by meaning, and
the hybrid that sfind uses by default.

Queries are grouped by how much they share with their answer, worked out from the index rather than
taken on trust: "exact" queries use the answer's own words; of the others, "some words" share at least
one indexed word with the answer (its file name included) and "no words" share none; "other language"
queries are in the other language. A result that sfind would call weak ("no strong match") counts as a
miss, since sfind reports nothing found. The queries in HELDOUT were never used to choose any setting,
and are reported apart. Nothing is written outside a temporary folder.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DOCS, HELDOUT, QUERIES  # noqa: E402

from semantic_find.embed import DEFAULT_MODEL, load  # noqa: E402
from semantic_find.index import Index  # noqa: E402
from semantic_find.text import index_terms, query_terms  # noqa: E402

KINDS = ("exact", "some words", "no words", "other language")


def kind_of(query: str, answer: str, labelled: str) -> str:
    if labelled == "words":
        return "exact"
    if labelled == "cross":
        return "other language"
    indexed = set(index_terms(DOCS[answer])) | set(index_terms(answer.replace("/", " ")))
    return "some words" if set(query_terms(query)) & indexed else "no words"


def evaluate(idx: Index, model, queries: list[tuple[str, str, str]]):
    modes = ("keyword", "meaning", "hybrid")
    table = {(m, k): [] for m in modes for k in (*KINDS, "all")}
    misses = []
    for query, answer, labelled in queries:
        kind = kind_of(query, answer, labelled)
        for mode in modes:
            hits = [h for h in idx.search(query, model, k=10, mode=mode) if not h.weak]  # weak: sfind says nothing found
            paths = [h.path for h in hits]
            rank = paths.index(answer) + 1 if answer in paths else None
            for key in (kind, "all"):
                table[(mode, key)].append(rank)
            if mode == "hybrid" and rank != 1:
                misses.append((query, answer, kind, rank, paths[:3]))
    return modes, table, misses


def show(title: str, modes, table) -> None:
    counts = {k: len(table[("hybrid", k)]) for k in KINDS}
    print(f"{title}: " + ", ".join(f"{counts[k]} {k}" for k in KINDS))
    print(f"{'':10}" + "".join(f"{k:>24}" for k in (*KINDS, "all")))
    print(f"{'':10}" + "".join(f"{'hit@1  hit@5   MRR':>24}" for _ in (*KINDS, "all")))
    for mode in modes:
        row = f"{mode:10}"
        for k in (*KINDS, "all"):
            ranks = table[(mode, k)]
            n = len(ranks) or 1
            h1 = sum(1 for r in ranks if r == 1) / n
            h5 = sum(1 for r in ranks if r and r <= 5) / n
            mrr = sum(1 / r for r in ranks if r) / n
            row += f"{h1:>10.0%}{h5:>7.0%}{mrr:>7.2f}" if ranks else f"{'-':>24}"
        print(row)
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SFIND_MODEL", DEFAULT_MODEL))
    ap.add_argument("--show", action="store_true", help="print every query that misses its answer in the hybrid mode")
    ap.add_argument("--haystack", type=int, default=0, metavar="N",
                    help="hide the documents among N generated filler files (eval/make_corpus.py), as in a real folder")
    args = ap.parse_args()
    model = load(args.model)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "corpus"
        if args.haystack:
            import subprocess

            subprocess.run([sys.executable, str(Path(__file__).with_name("make_corpus.py")), str(root / "filler"),
                            "--files", str(args.haystack), "--seed", "7"], check=True, capture_output=True)
        for rel, text in DOCS.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        idx = Index(root, Path(tmp) / "index.sqlite")
        idx.update(model, args.model)
        where = f" among {args.haystack} filler files" if args.haystack else ""
        print(f"model: {args.model}; {len(DOCS)} documents{where}\n")
        modes, table, misses = evaluate(idx, model, QUERIES)
        show(f"{len(QUERIES)} queries (the settings were chosen on these)", modes, table)
        modes, held, held_misses = evaluate(idx, model, HELDOUT)
        show(f"{len(HELDOUT)} held-out queries (never used to choose anything)", modes, held)
        idx.close()
    if args.show:
        for title, m in (("tuning set", misses), ("held out", held_misses)):
            print(f"hybrid misses, {title} (query → expected; rank; first three):")
            for q, a, kind, rank, top in m:
                print(f"  [{kind}] {q} → {a}; {rank or '-'}; {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
