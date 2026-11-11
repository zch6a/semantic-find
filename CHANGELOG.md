# Changelog

## 0.1.0 (2026-11-12)

First version.

- `sfind QUERY [FOLDER...]`: indexes a folder the first time and searches it; later runs read only
  changed files. Results show the best passage of each file, with why it matched.
- Search by meaning with a static embedding model (Model2Vec's inference reimplemented with numpy and
  `tokenizers`; no PyTorch), and by keywords with SQLite FTS5 (identifiers split into parts, CJK by
  characters and character pairs). The two are combined by score, measured on `eval/`.
- Built-in models pinned by revision, size and SHA-256; downloaded once, with consent, resumable,
  from Hugging Face or a mirror (`HF_ENDPOINT`). Ollama embedding models on this machine.
- Reads text, code, Markdown, HTML, Word, PowerPoint, Excel, OpenDocument, EPUB, notebooks and (with
  the `pdf` extra) PDF, in UTF-8, UTF-16 and common legacy encodings.
- In git repositories, follows what git tracks and ignores.
- Shows only what matches: weak matches after a strong one are dropped, a match by meaning must stand out
  from the rest of a large folder, so a search for something absent finds nothing; copies of a passage are
  shown once.
- `sfind index`, `status`, `forget`, `models`; `-g`, `-l`, `--json`, `--mode`.
- For contributors: `eval/` (an evaluation set with held-out queries, a corpus maker, the stand-out study) and
  `scripts/mutate.py`.
