<div align="center">

# semantic-find

**Search your files by what they mean, offline.** `sfind` is grep for meaning: type what you are looking
for in your own words, in English, Chinese or any of 100 other languages, and it finds the passage in
your notes, documents and code, even when the words are different.

<a href="https://github.com/zch6a/semantic-find/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zch6a/semantic-find/actions/workflows/ci.yml/badge.svg"></a>
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
<img alt="Offline" src="https://img.shields.io/badge/your%20files-stay%20local-2ea44f">
<a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue"></a>

**English** · [简体中文](README.zh-CN.md)

</div>

---

```console
$ sfind "how long does getting a new travel document take" ~/notes
notes/travel.md:1
    Travel — Passport renewal: book an appointment at the post office, bring the old passport
    and two photos. It takes about six weeks.
    (words: travel · meaning 0.64 · 1 more in this file)

$ sfind "exponential backoff when a request fails" src
src/net.py:1
    def fetch(url, retries=3): # try again after a pause that doubles each time …
    (meaning 0.42)
```

Neither "passport" nor "backoff" was in the query or the file: the match is by meaning.

## What it does

- **No setup.** `sfind QUERY [FOLDER]` indexes the folder the first time and searches it. Later
  searches read again only the files that changed, so they start at once.
- **Words and meaning together.** Each passage is scored by meaning (embedding similarity) and by how
  much of the query's words it contains (rarer words count more). An exact name such as
  `fetchWithRetry` is found as surely as with grep, and a question in plain words finds the answer that
  uses other words. Every result says why it matched. [How well it works](#how-well-it-finds-things)
  is measured, not guessed.
- **Chinese, and every language.** The default model covers 101 languages, and keyword search reads
  Chinese, Japanese and Korean by characters and character pairs, so `sfind "数据库连接池怎么配置"`
  works with no dictionary. A query often finds a file in another language too ("what tires for snow" finds
  "记得把车的轮胎换成雪地胎"), though less reliably than in the same language.
- **Your documents, not just code.** Markdown, plain text and code in any language, HTML, Word (.docx),
  PowerPoint (.pptx), Excel (.xlsx), OpenDocument, EPUB, Jupyter notebooks, and PDF (with
  `pip install "semantic-find[pdf]"`). Files in older encodings (GB18030, Big5, Shift-JIS, EUC-KR,
  Windows-1252) are recognized and read. A damaged file is skipped, never the folder.
- **Small and fast.** The model is a static embedding model ([Model2Vec](https://github.com/MinishLab/model2vec)):
  no neural network runs, so there is no PyTorch and no GPU. sfind itself needs numpy and Hugging
  Face's `tokenizers`; `tokenizers` brings `huggingface_hub` and its own dependencies, about 20 small
  packages in all.
- **Says how sure it is.** Weak matches after a strong one are left out. In a folder of 500 passages
  or more, when nothing matches strongly (a search for something that is not in the folder, say),
  sfind says so and shows the three closest passages as such, exiting with 1 ("nothing found") for
  scripts; a small folder has too little for a match to stand out from, so there the closest passages
  are simply listed. In a very large folder (tens of thousands of passages) the best of many vaguely
  similar passages can stand out by chance, so a search for something absent may still show a few. A passage with every word of the query is always shown. Copies of the same text
  are shown once ("the same text in 24 other files").
- **Grep-like.** `-g "*.md"` for some files only, `-l` for file names only (of the best `-k`, 10 by
  default, and their copies), `--json` for scripts, exit code 1 when nothing is found and 2 on an error.

## Privacy

- **Nothing leaves your machine.** The index is a SQLite file in your user data folder
  (`sfind status` says where), never in the folder searched. Searching needs no network.
- **The model is downloaded once, and only when you agree.** `sfind` asks before downloading it (or run
  `sfind models download`). Every file of a built-in model is pinned in the code, by revision, size
  and SHA-256, so a mirror cannot change it. Set `HF_ENDPOINT` to download from a mirror instead, for
  example `HF_ENDPOINT=https://hf-mirror.com`.
- Until a model is downloaded, `sfind` searches by keywords and says so.
- With `--model ollama:<name>`, passages go to Ollama, and only to an Ollama server on this machine
  (never through a proxy, never to a cloud model).
- The index holds the text of your files, so on Linux and macOS it is readable by you only.
  `sfind forget FOLDER` deletes it.

## Install

```bash
pip install "semantic-find @ git+https://github.com/zch6a/semantic-find"
```

or `uv tool install git+https://github.com/zch6a/semantic-find` (or `pipx install ...`) to have the
`sfind` command without touching other Python projects. Add `[pdf]` for PDF files.

## Use

```bash
sfind "renew passport" ~/Documents         # search a folder (indexed the first time)
sfind "数据库连接池怎么配置" .                 # Chinese, as well as English
sfind "retry with backoff" src -g "*.py"    # only some files
sfind "budget for 2026" ~/Documents -k 20 -a   # 20 results, every matching passage
sfind "invoice" ~/Documents -l               # only the paths, for piping
sfind "invoice" ~/Documents --json           # for scripts
sfind "fetchWithRetry" --mode keyword        # exact names only
sfind index ~/notes                          # index without searching
sfind status ~/notes                         # what is indexed, and where
sfind forget ~/notes                         # delete the index
sfind models                                 # the models, and which are downloaded
```

In a git repository, the files git tracks are searched, and the untracked ones it does not ignore,
except dependency and build folders (`node_modules`, `.venv`, `dist` and so on). Elsewhere those folders
and hidden folders are left out. A `.sfindignore` file in the folder can list more patterns. Two
`sfind` runs at once on the same folder are fine: one updates the index, the other searches it as it
is.

### Models

| Model | Languages | Download | |
|---|---|---|---|
| `potion-multilingual-128M` (default) | 101, Chinese and English among them | 506 MB | one model for documents in any language |
| `potion-retrieval-32M` | English | 125 MB | trained for search |
| `potion-base-8M` | English | 29 MB | the smallest; for a quick try |
| `ollama:<name>` | the model's | via Ollama | for example `ollama:bge-m3` or `ollama:qwen3-embedding`: slower, more accurate |

`--model` or `SFIND_MODEL` chooses one. A folder holding a Model2Vec model also works.

## How it works

1. **Files** are listed (`git ls-files` in a repository), and only new or changed ones (by size and
   modification time) are read.
2. **Passages**: text is cut at paragraphs into passages of about 700 characters, never more than
   1,500. A Markdown heading starts a new one and is kept as its context.
3. **Keywords**: each passage's words go into an SQLite FTS5 table. Identifiers are also split into
   their parts (`fetchWithRetry` → fetch, with, retry), and CJK text into characters and character
   pairs.
4. **Meaning**: each passage, with its file name and heading, becomes a vector: the average of its
   tokens' vectors in the static model, stored as float16.
5. **Search**: each candidate scores its similarity to the query (relative to the best one) plus 0.15
   times the share of the query's words it contains, rarer words weighing more and a word that is in
   no passage at all weighing most (so "xylophone lessons" is not covered by "lessons"). When several
   folders are searched together, all are scored against the best match in any of them. This is a
   convex combination of normalized scores, which Bruch, Gai and Ingber found better than rank fusion
   ("An Analysis of Fusion Functions for Hybrid Retrieval", ACM TOIS 2023). Reciprocal-rank
   fusion, the usual way to combine two rankings, was tried first and measured worse: it let a passage
   sharing one word of four outrank the one that meant what was asked.
6. **What is shown**: a passage found only by meaning must be at least half as close as the best one.
   In a folder of 500 passages or more it counts as a strong match only if it stands out from the rest,
   at least three standard deviations above the folder's mean similarity to the query; otherwise it is
   weak, and weak matches are shown only when there is nothing strong. They are labelled rather than
   dropped because nothing tells them apart reliably: measured on up to 64,000 passages
   (`eval/standout.py`), the weakest tenth of right answers look just like the best matches for
   subjects absent from the folder. Results scoring under 60% of the best are dropped, the best passage
   of each file is shown, and copies of a passage are shown once.


## How well it finds things

Measured with `eval/evaluate.py` on `eval/data.py`: 53 synthetic notes, documents and code files in
English and Chinese, some of them near-misses for others, and 130 queries with a known answer. The
evaluation sorts the queries itself, from the index: **exact** queries use the answer's own words,
**some words** queries share at least one word with it (its file name included), **no words** queries
share none, and **other language** queries are in the other language. A result that sfind would call
weak counts as a miss. "Top 1" is how often the right file comes first.

| Hybrid (the default), default model | exact | some words | no words | other language | all, top 1 | all, top 5 |
|---|---|---|---|---|---|---|
| 100 queries the settings were chosen on | 100% | 93% | 54% | 70% | 78% | 91% |
| 30 held-out queries, never used to choose anything | 75% | 100% | (1 query) | 40% | 87% | 93% |
| the 100, with the documents among 3,000 other files | 100% | 71% | 50% | 40% | 62% | 68% |

Keyword search alone gets the exact queries and 82% of the "some words" ones, and nothing else; search
by meaning alone gets 74% of the 100 first. With the English-only `potion-base-8M`, hybrid search finds
50% of the 100 first, and nothing across languages, as expected. The third row is closer to a real
folder: the same documents hidden among 3,000 generated notes, Office files, PDFs and code on similar
subjects.

What this means in use: exact names and words are found as with grep; a question in other words
finds the right file first about half the time and among the first five most of the time; across
languages it is the weak point of a static model. The held-out queries turned out to share more words
with their answers than meant (20 of 30), so they are an easier test, not a fairer one; the set is
small. Take these numbers as a sanity check, not a benchmark.

## Status

0.1, in development. Tested with synthetic notes and a tiny hand-made model (no download in the
tests), by hand on the Python standard library (728 files and 12,254 passages indexed in 5 s on a
laptop), on a generated folder of 3,060 mixed files (`eval/make_corpus.py`; indexed in 2.5 s once
the files are in the disk cache) and on one of 6,060 longer files: 76,915 passages indexed in 49 s
into a 195 MB index, searched in about 2 s (start-up included). On Windows, files written moments before are slow to read the first
time while the antivirus scans them: 35 s for those 3,060 files. A search takes about 2 s with the multilingual model, 1 s of which is loading its 18 MB
tokenizer, and under a second with the English ones.

## Develop

```bash
uv venv && uv pip install -e ".[pdf]" pytest ruff
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
