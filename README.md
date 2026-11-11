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
  no neural network runs, so there is no PyTorch and no GPU. The only dependencies are numpy and
  Hugging Face's `tokenizers`.
- **Only what matches.** A search for something that is not in the folder finds nothing, rather than
  the least bad passages. Weak matches after a strong one are left out, and copies of the same text are
  shown once ("the same text in 24 other files").
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
   times the share of the query's words it contains, rarer words weighing more. Reciprocal-rank
   fusion, the usual way to combine two rankings, was tried first and measured worse: it let a passage
   sharing one word of four outrank the one that meant what was asked.
6. **What is shown**: a passage found only by meaning must be at least half as close as the best one,
   and in a folder of 500 passages or more it must stand out from the rest: at least three standard
   deviations above the folder's mean similarity to the query. Measured on 3,000 files, right
   answers stood 3.6 to 7.9 deviations out, and the best matches for subjects absent from the folder
   at most 2.8. Results scoring under 60% of the best are dropped, the best passage of each file is
   shown, and copies of a passage are shown once.


## How well it finds things

Measured with `eval/evaluate.py` on `eval/data.py`: 53 synthetic notes, documents and code files in
English and Chinese, some of them near-misses for others, and 100 queries with a known answer. 18
use the answer's own words, 52 say it in other words (sharing no content word with it), and 30 are
in the other language. "Top 1" is how often the right file comes first.

| Default model | exact words, top 1 | other words, top 1 | other language, top 1 | all, top 1 | all, top 5 |
|---|---|---|---|---|---|
| keyword (`--mode keyword`) | 100% | 44% | 0% | 41% | 45% |
| meaning (`--mode meaning`) | 89% | 71% | 70% | 74% | 91% |
| **hybrid** (the default) | **100%** | **73%** | 67% | **76%** | **91%** |
| hybrid, the documents among 3,000 other files (`--haystack 3000`) | 100% | 67% | 37% | 64% | 75% |

The last row is closer to a real folder: the same documents hidden among 3,000 generated notes,
Office files, PDFs and code on similar subjects. With the English-only `potion-base-8M`, hybrid search
finds 52% first (53% among the 3,000), and nothing across languages, as expected.

What this means in use: exact names and words are found as with grep; a question in other words
finds the right file first about three times in four in a small folder, and two in three in a large
one; across languages it often works but is the weak point of a static model. The set is small, so
take it as a sanity check, not a benchmark.

## Status

0.1, in development. Tested with synthetic notes and a tiny hand-made model (no download in the
tests), by hand on the Python standard library (728 files and 12,254 passages indexed in 5 s on a
laptop) and on a generated folder of 3,060 mixed files (`eval/make_corpus.py`; indexed in 2.5 s once
the files are in the disk cache). On Windows, files written moments before are slow to read the first
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
