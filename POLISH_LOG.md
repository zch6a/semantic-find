# Polish log

## Development

**What it is for.** In 2026 there are many semantic-search tools for code, most built for coding agents (grepai, embgrep, vecgrep, SIFS and others). Most need Ollama or an API key, and most ship one English model. semantic-find is for a person's own files: notes, Office documents, PDFs and code, in Chinese as much as in English. It works the way grep does: `sfind QUERY [FOLDER]`, with no setup step.

**Design decisions**
- **Model**: a static embedding model (Model2Vec). Its inference was reimplemented in about 60 lines: a safetensors reader in numpy, Hugging Face's `tokenizers`, and the mapping, weights and normalization the model asks for. Checked against the `model2vec` package itself, the vectors match exactly (largest difference 0.0). The only dependencies are numpy and tokenizers, so there is no PyTorch and no GPU. The default is `potion-multilingual-128M` (101 languages, 507 MB, downloaded once). Ollama is optional, on this machine only.
- **Hybrid search**: BM25 through SQLite FTS5 on pre-tokenized terms, and cosine similarity over float16 vectors, fused with RRF (k = 60). Keyword terms:
  - identifiers are split into their parts (`fetchWithRetry` → fetch, with, retry);
  - CJK runs become overlapping character pairs, so Chinese works without a dictionary;
  - filler words (the, how, 怎么, 什么…) are dropped from queries.
- **Noise floor**: a similarity ranking always has a "best", so a passage found only by meaning must reach half the best similarity, and at least 0.1. A search for something absent finds nothing.
- **Index**: one SQLite file per folder in the user data folder, never inside the searched folder. Files are re-read only when their size or modification time changes. A new model re-embeds everything.
- **Files**: `git ls-files` in a repository, so .gitignore is honoured as git reads it. Elsewhere well-known clutter and hidden folders are skipped, and `.sfindignore` can add patterns. Text formats:
  - .docx, .pptx and .xlsx, ODF and EPUB, read with the standard library;
  - HTML and notebooks;
  - PDF through the optional `pypdf`;
  - GB18030 and UTF-16 text.
- **Privacy**: the model is downloaded only with consent (asked on a terminal; `sfind models download` otherwise), resumable, and checked against Hugging Face's size and SHA-256. `HF_ENDPOINT` mirrors are supported. Until a model is there, search is by keywords, and sfind says so.

**Found while building, by using it**
- Filler words matched ("words: a"), and a semantic ranking listed every file. Hence the stop words and the noise floor.
- A Python `# comment` was taken as a Markdown heading. Headings now count only in formats that have them.
- A missing folder in `sfind passport /no/such/dir` became part of the query. Arguments that look like paths are now treated as paths.
- PDF text puts form feeds mid-line, and pages were miscounted.
- Matched Chinese pairs were shown as "连接, 接池". They are now joined back into "连接池".

**Measured by hand**
- Python's standard library (728 files, 12,254 passages): indexed in 17 s with `potion-base-8M`, and "parse an email address into name and domain" finds `email/utils.py`. A search, including the check for changed files, takes about 1.2 s.
- Synthetic Chinese and English notes with the default model: Chinese queries find Chinese notes (similarity 0.4–0.8), and "what tires for snow" finds a Chinese note about snow tyres. One cross-language query ranked a less fitting note first, and the README says that cross-language matching is less reliable.

**Open for the polishing rounds**
- A model download started in a hidden background process sat at 0 bytes for minutes, while the same command in the foreground finished in 61 s. Not explained yet. Without a terminal there is no progress either, so a stalled download is silent.
- Indexing at 12k passages in 17 s: the embedding should take about a second, so the time is in reading and SQLite. Profile it.
- An evaluation set (queries with known answers, in both languages) to measure hybrid against keyword-only and meaning-only, for the README.
- PDF extraction is untested (it needs a PDF fixture with text).

34 tests (tiny hand-made model; no download, no network), ruff clean.