"""How well each search mode finds the right file, on the evaluation set in eval/data.py.

    python eval/evaluate.py                       # the default model (it must be downloaded)
    python eval/evaluate.py --model potion-base-8M
    python eval/evaluate.py --model ollama:bge-m3
    python eval/evaluate.py --haystack 3000       # the documents hidden among 3,000 generated files

For every query it reports whether the right file is first (hit@1), among the first five (hit@5), and the
mean reciprocal rank (MRR: 1 for first, 1/2 for second, ...), for keyword search, search by meaning, and
the hybrid that sfind uses by default, split by kind of query. Nothing is written outside a temporary folder.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DOCS, QUERIES  # noqa: E402

from semantic_find.embed import DEFAULT_MODEL, load  # noqa: E402
from semantic_find.index import Index  # noqa: E402


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
        modes = ("keyword", "meaning", "hybrid")
        kinds = ("words", "meaning", "cross")
        table = {(m, k): [] for m in modes for k in (*kinds, "all")}
        misses = []
        for query, answer, kind in QUERIES:
            for mode in modes:
                paths = [h.path for h in idx.search(query, model, k=10, mode=mode)]
                rank = paths.index(answer) + 1 if answer in paths else None
                for key in (kind, "all"):
                    table[(mode, key)].append(rank)
                if mode == "hybrid" and rank != 1:
                    misses.append((query, answer, kind, rank, paths[:3]))
        idx.close()
    print(f"model: {args.model}; {len(DOCS)} documents" + (f" among {args.haystack} filler files" if args.haystack else "") + f", {len(QUERIES)} queries "
          f"({', '.join(f'{sum(1 for q in QUERIES if q[2] == k)} {k}' for k in kinds)})\n")
    print(f"{'':10}" + "".join(f"{k:>24}" for k in (*kinds, "all")))
    print(f"{'':10}" + "".join(f"{'hit@1  hit@5   MRR':>24}" for _ in (*kinds, "all")))
    for mode in modes:
        row = f"{mode:10}"
        for k in (*kinds, "all"):
            ranks = table[(mode, k)]
            n = len(ranks) or 1
            h1 = sum(1 for r in ranks if r == 1) / n
            h5 = sum(1 for r in ranks if r and r <= 5) / n
            mrr = sum(1 / r for r in ranks if r) / n
            row += f"{h1:>10.0%}{h5:>7.0%}{mrr:>7.2f}"
        print(row)
    if args.show and misses:
        print("\nhybrid misses (query → expected; rank; first three):")
        for q, a, kind, rank, top in misses:
            print(f"  [{kind}] {q} → {a}; {rank or '-'}; {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
