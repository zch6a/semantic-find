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
## Round 1

Three things this round: an independent review of the code (with every finding reproduced), the open items from development, and a measurement of search quality.

**Search quality, measured.** `eval/` is a new evaluation set: 29 synthetic notes, documents and code files in English and Chinese (with near-misses, such as two notes about passports), and 45 queries with known answers, of three kinds (same words, other words, other language). `eval/evaluate.py` reports hit@1, hit@5 and MRR per mode.
- **Reciprocal-rank fusion, the usual way to merge rankings, was the weak point.** With the default model it found the right file first for only 22% of cross-language queries, against 67% for search by meaning alone. With k = 60, rank 4 counts almost as much as rank 1, so a passage sharing one word of four with the query ("settings" in "database connection pool settings") outranked the Chinese note that meant what was asked. Damping keyword hits by coverage did not help, because the problem is the ranks themselves.
- **It was replaced by score fusion**: similarity relative to the best, plus 0.5 × the share of the query's words the passage contains, IDF-weighted, over terms that occur in the index. The weight 0.5 was chosen from 0.2–1.5 on three models. With the default model:

  | | top 1 | top 1, same language, other words | top 1, other language |
  |---|---|---|---|
  | rank fusion | 80% | 93% | 22% |
  | score fusion | 84% | 93% | 44% |

  English-only models are within one query either way. Searching by meaning alone is still better across languages (67%), and the README says so.

**The review: 9 confirmed findings and 5 suspected. All are fixed, each with a test in `tests/test_review.py`.**
- **One damaged file stopped the folder for good.** An empty, truncated or password-protected PDF, or a notebook with `"cells": null`, raised out of `update`. The update was one transaction, so nothing was saved and every later run failed on the same file. It also exited with 1, which means "nothing found". Now a file that cannot be read is skipped, progress is committed every 200 files (an interrupted first index keeps what it has done), and any unexpected error exits 2 with a one-line message.
- **Two sfinds at once**: the second died with "database is locked". An OS file lock now lets one update at a time; the other searches the index as it is and says so.
- **A git merge conflict** lists a file once per side, which broke a UNIQUE constraint. Duplicates are dropped.
- **Zip bombs**: a 299 KB .docx unpacked to 300 MB and peaked at 1.1 GB of memory. Unpacking is now capped at 200 MB per document, checked against each part's declared size before reading, and at 5,000 parts.
- **`-g` missed matches** beyond the first 300 candidates. The pattern now filters before ranking, in SQL for keywords and as a mask for vectors.
- **A download whose `.part` was already whole** (interrupted during the hash check) asked for a range past the end and got 416, forever. Now:
  - a whole part is only checked;
  - one that is too long is discarded;
  - a 416 deletes the part.
- **Piped `-l` and `--json` on Windows** turned "café 여권 ✓" into "?". Output into a pipe is now UTF-8.
- **Legacy encodings**: cp1252, Shift-JIS and EUC-KR files were read as GB18030 mojibake or dropped. charset-normalizer was tried and got short GB18030 Chinese wrong (as Korean) and every cp1252 sample. Instead, each candidate decoding is scored by its share of everyday characters in its own script: GB2312 level-1 hanzi, Big5 frequent hanzi, kana and JIS level-1 kanji, Hangul, Latin letters. EUC-KR and GB18030 read the same bytes as equally common characters, so Korean's spaces between words break the tie. 14 of 15 samples come out right; the miss is a string of rare characters only.
- **Smaller findings:**
  - a query word that named a folder ("how tests work") became a path; paths are now only the trailing arguments;
  - tracked files under `build/` and `dist/` were skipped in git repositories (git's view now wins for tracked files, and untracked dependency folders are still skipped);
  - `--no-update` with another model gave a numpy error, and is now explained;
  - index names ignored case on Linux;
  - `OLLAMA_HOST=0.0.0.0` was refused;
  - a one-character Chinese query never matched (single CJK characters are now indexed too).
- **Suspected, also fixed:**
  - the download trusted the mirror for both the file and its hash. Every file of the built-in models is now pinned by revision, size and SHA-256, and the Hugging Face API is no longer called at all. During this round it answered "maximum queue size reached", so that also makes downloads more reliable;
  - on Linux and macOS, indexes are readable by their owner only (0600 files, 0700 folder);
  - Office text kept tabs and line breaks, and text boxes no longer repeat their paragraphs;
  - typed-in Excel cells are read;
  - crafted safetensors headers (offsets outside the file) are refused.
- Checked by the reviewer and found fine:
  - FTS5 query syntax in queries;
  - queries of only punctuation or stop words;
  - files deleted while indexing;
  - non-ASCII names in git mode;
  - Ollama requests bypassing proxies.

**Open items from development**
- **The stalled download was a measurement error.** On Windows, a directory listing shows a stale size for a file that is still open, and opening a handle showed it growing: 82, 92, 102 MB. The real flaw was silence: without a terminal there was no progress at all. Now a line is printed at each tenth.
- **Indexing speed**: profiling showed that splitting words into terms took 5.3 s of 9.8 s, because the regexes ran on every token. Terms are now cached per distinct word. The standard library (728 files, 12,254 passages) indexes in 5.0 s, against 6.2 s before with a warm disk cache; the 17 s measured earlier had a cold one.
- **Search start-up**: the multilingual model took 1.64 s to load, and 0.7 s of that was spent working out the unknown-token id and median token length on every run. Both are now cached next to the model, so it loads in 0.94 s. Most of what is left is the tokenizer parsing its 18 MB JSON. A search takes about 1.9 s in all with the default model and 0.7 s with an English one, and the README says so. `urllib.request` is now imported only to download or to reach Ollama.
- **PDF**: tested with a PDF written by hand (`tests/pdfmaker.py`). The test found that a broken PDF raised pypdf's own error type, which was the review's first finding seen from another side.

**Docs**: the README has a measured-results table in both languages, the scoring described as it is, git and concurrency behaviour, the pinned downloads, and timings that are now measured. New CONTRIBUTING.md and CHANGELOG.md.

60 tests, ruff clean (CI now lints `eval/` too).

## Round 2

Real use. `eval/make_corpus.py` (new) writes a large, messy synthetic folder: 3,060 files of notes, GB18030 text, .docx, PDFs and code, in English and Chinese. It has 50 copied notes, clutter that should be skipped, damaged and empty files, a note twelve folders deep, and five planted documents with known content. Every command and option was tried on it with the default model, the way a new user would.

**What real use showed, and what changed**
- **Weak matches after a strong one**: "when should the timing belt be checked" found the service record first, then two unrelated notes. A result must now score at least 60% of the best one (`SCORE_FLOOR`). On eval/ this costs nothing in hit@5 and cuts results per query from 4.3 to 2.3.
- **Searches for what is not there returned junk.** "bcrypt", "xylophone lessons" and "量子计算" are absent from the folder, yet each listed five unrelated notes at similarity 0.25–0.3, because relative floors always keep something. Measured on the folder, right answers stand 3.6 to 7.9 standard deviations above the mean similarity of all passages to the query, and noise at most 2.8. A passage found only by meaning must now stand out by 3 deviations. All three searches now say "nothing found", and the planted answers come back alone.

  In a small folder there is no background to stand out from: on eval/ alone the test cost 18 points of hit@5. So it applies only from 500 passages. A limit, noted for later: the largest of n noise values grows like √(2 ln n), so very large folders may need a higher threshold. On 4,000 passages the noise stayed at 2.8.
- **Copies**: "backup every night" listed three identical copied notes. Exact copies of a passage are now shown once, with "the same text in 24 other files". `-l` still lists them all, and `--json` has them in `same`.
- **`-l` printed absolute paths while normal output was relative.** Both are now relative to the current folder, as grep prints them.
- **Snippets were cut at 240 characters**, so Chinese snippets were twice as wide on screen. They are now cut by display width.
- **pypdf's own warnings** ("EOF marker not found") leaked into the output; they are silenced. Also: "1 passages embedded" is now singular.
- **First index of the folder: 37.5 s.** The cause is not sfind: plain reads of 3,061 freshly written files took 35.2 s the first time and 0.37 s the second (the antivirus scans new files). Warm, the folder indexes in 2.5 s with the default model, which embeds 2,000 passages in 0.1 s. The README mentions it.
- **Fine as it was:**
  - deleted and edited files are picked up on the next search (0.3 s);
  - searching from inside the folder with no path;
  - cross-language finds ("how often do I vaccinate the cat" → the Chinese note about the cat);
  - `-g`, `-a`, `status`.

**The evaluation set, made honest.** Keyword search alone answered 82% of the old "other words" queries, because they shared words with their answers. There are now 23 more documents and 55 more queries written to share no content word with the answer (100 queries in all: 18 exact, 52 other words, 30 other language). Keyword search now gets 44% of the other-words queries. `--haystack N` hides the documents among N generated files.

**That exposed a regression: hybrid scored below meaning alone** (71% top 1 against 74%), losing on cross-language queries. The keyword weight was swept from 0 to 0.5 on three models, and a coverage exponent was tried and made no difference. At 0.15 the default model reaches 76% top 1 and 91% top 5 (meaning alone: 74% and 91%), with exact names still at 100%. Below 0.15 the default model starts missing exact names ("bcrypt", "TokenBucket"). With the English models, 0.15 is within one or two queries of their best. The score floor was checked again at the new weight; 0.6 still costs nothing.

| default model, hybrid | top 1 | top 5 | other words, top 1 | other language, top 1 |
|---|---|---|---|---|
| eval/ alone | 76% | 91% | 73% | 67% |
| among 3,000 other files | 64% | 75% | 67% | 37% |

66 tests (new: `tests/test_results.py` for copies, the score floor, the stand-out test, singular and plural, and the corpus maker), ruff clean.
