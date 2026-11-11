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
- **Words and meaning together.** Each passage is ranked by keywords (BM25) and by meaning (embedding
  similarity), and the two rankings are fused (reciprocal-rank fusion). An exact name such as
  `fetchWithRetry` is found as surely as with grep, and a question in plain words finds the answer that
  uses other words. Every result says why it matched.
- **Chinese, and every language.** The default model covers 101 languages, and keyword search reads
  Chinese, Japanese and Korean by character pairs, so `sfind "数据库连接池怎么配置"` works with no
  dictionary. A query often finds a file in another language too ("what tires for snow" finds
  "记得把车的轮胎换成雪地胎"), though less reliably than in the same language.
- **Your documents, not just code.** Markdown, plain text and code in any language, HTML, Word (.docx),
  PowerPoint (.pptx), Excel (.xlsx), OpenDocument, EPUB, Jupyter notebooks, and PDF (with
  `pip install "semantic-find[pdf]"`). GB18030 and UTF-16 files are read too.
- **Small and fast.** The model is a static embedding model ([Model2Vec](https://github.com/MinishLab/model2vec)):
  no neural network runs, so there is no PyTorch and no GPU. The only dependencies are numpy and
  Hugging Face's `tokenizers`.
- **Grep-like.** `-g "*.md"` for some files only, `-l` for file names only, `--json` for scripts,
  exit code 1 when nothing is found.

## Privacy

- **Nothing leaves your machine.** The index is a SQLite file in your user data folder
  (`sfind status` says where), never in the folder searched. Searching needs no network.
- **The model is downloaded once, and only when you agree.** `sfind` asks before downloading it (or run
  `sfind models download`), and checks it against the size and SHA-256 that Hugging Face lists. Set
  `HF_ENDPOINT` to use a mirror, for example `HF_ENDPOINT=https://hf-mirror.com`.
- Until a model is downloaded, `sfind` searches by keywords and says so.
- With `--model ollama:<name>`, passages go to Ollama, and only to an Ollama server on this machine
  (never through a proxy, never to a cloud model).
- The index holds the text of your files. `sfind forget FOLDER` deletes it.

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

In a git repository, files that git ignores are not searched. Elsewhere, version-control, dependency
and build folders (`node_modules`, `.venv`, `dist` and so on) and hidden folders are left out. A
`.sfindignore` file in the folder can list more patterns.

### Models

| Model | Languages | Download | |
|---|---|---|---|
| `potion-multilingual-128M` (default) | 101, Chinese and English among them | 507 MB | one model for documents in any language |
| `potion-retrieval-32M` | English | 125 MB | trained for search: better for English-only files |
| `potion-base-8M` | English | 30 MB | the smallest; for a quick try |
| `ollama:<name>` | the model's | via Ollama | for example `ollama:bge-m3` or `ollama:qwen3-embedding`: slower, more accurate |

`--model` or `SFIND_MODEL` chooses one. A folder holding a Model2Vec model also works.

## How it works

1. **Files** are listed (`git ls-files` in a repository), and only new or changed ones (by size and
   modification time) are read.
2. **Passages**: text is cut at paragraphs into passages of about 700 characters, never more than
   1,500. A Markdown heading starts a new one and is kept as its context.
3. **Keywords**: each passage's words go into an SQLite FTS5 table. Identifiers are also split into
   their parts (`fetchWithRetry` → fetch, with, retry), and CJK text into character pairs.
4. **Meaning**: each passage, with its file name and heading, becomes a vector: the average of its
   tokens' vectors in the static model, stored as float16.
5. **Search**: the query is ranked both ways. A passage found only by meaning must be at least half as
   close as the best one, so a search for something that is not there finds nothing rather than
   noise. The two rankings are fused, and the best passage of each file is shown.

## Status

0.1, in development. Tested with synthetic notes and a tiny hand-made model (no download in the tests),
and by hand on the Python standard library (728 files, 12,254 passages indexed in 17 s on a laptop,
searches in about a second).

## Develop

```bash
uv venv && uv pip install -e ".[pdf]" pytest ruff
uv run pytest
```

## License

[MIT](LICENSE)
