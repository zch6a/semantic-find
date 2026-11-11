# Changelog

## 0.1.0 (unreleased)

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
- `sfind index`, `status`, `forget`, `models`; `-g`, `-l`, `--json`, `--mode`.
