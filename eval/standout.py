"""How the stand-out test behaves as a folder grows: for folders of 1,000 to 64,000 generated passages
(with the eval documents among them), how far the right answer stands above the mean (z), and how far the
best match for a subject that is not in the folder at all does. Passages are embedded in memory; nothing
is written to disk.

    python eval/standout.py [--model NAME] [--sizes 1000 4000 16000 64000]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DOCS, QUERIES  # noqa: E402
from make_corpus import paragraph  # noqa: E402

from semantic_find.embed import DEFAULT_MODEL, load  # noqa: E402

ABSENT = ["xylophone lessons", "sailing knots for beginners", "volcano eruption warning", "chess opening theory", "satellite orbit decay",
          "量子计算", "股票期权行权", "古筝考级", "火山喷发预警", "卫星轨道衰减"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SFIND_MODEL", DEFAULT_MODEL))
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 4000, 16000, 64000])
    args = ap.parse_args()
    model = load(args.model)
    rng = random.Random(5)
    biggest = max(args.sizes)
    filler = [paragraph(rng, rng.random() < 0.4)[1] for _ in range(biggest)]
    fill_vecs = model.encode(filler)
    doc_names = list(DOCS)
    doc_vecs = model.encode([f"{Path(n).stem}\n{DOCS[n]}" for n in doc_names])
    queries = [q for q, a, k in QUERIES if k != "words"]
    answers = [doc_names.index(a) for q, a, k in QUERIES if k != "words"]
    q_vecs = model.encode(queries)
    absent_vecs = model.encode(ABSENT)
    print(f"model {args.model}; z = (similarity - mean) / standard deviation over all passages\n")
    print(f"{'passages':>9} {'answer z: 10th pct':>19} {'median':>7} {'absent: max z':>14} {'mean of max':>12} {'√(2 ln n)':>10}")
    for n in args.sizes:
        mat = np.vstack([fill_vecs[:n], doc_vecs])
        sims = mat @ q_vecs.T  # passages × queries
        mu, sd = sims.mean(axis=0), sims.std(axis=0)
        z_answer = (sims[n + np.array(answers), np.arange(len(queries))] - mu) / sd
        a_sims = mat @ absent_vecs.T
        z_absent = (a_sims.max(axis=0) - a_sims.mean(axis=0)) / a_sims.std(axis=0)
        print(f"{n + len(doc_names):>9} {np.percentile(z_answer, 10):>19.2f} {np.median(z_answer):>7.2f} {z_absent.max():>14.2f} "
              f"{z_absent.mean():>12.2f} {np.sqrt(2 * np.log(n + len(doc_names))):>10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
