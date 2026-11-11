# Contributing

Thanks for helping. semantic-find is small on purpose: plain Python, numpy and Hugging Face's
`tokenizers`, with pypdf as an optional extra. The most useful things to send are a folder (of
synthetic files) where a search finds the wrong thing, and what you expected it to find.

## Set up

```bash
git clone https://github.com/zch6a/semantic-find && cd semantic-find
uv venv && uv pip install -e ".[pdf]" pytest ruff
uv run pytest                 # about 85 tests; no model download, no network
uv run ruff check src tests eval scripts
uv run python scripts/mutate.py   # plants each bug in its list in turn; every one must make a test fail
```

Set `SEMANTIC_FIND_HOME` to a temporary folder while you try things, so that test indexes and models do
not end up in your real data folder.

## How the tests work

- **A tiny hand-made model.** `tests/fakemodel.py` writes a Model2Vec-format model: a word-level
  tokenizer and vectors in which words of one topic point the same way. "Search by meaning" is
  tested with it, with no download.
- **Stand-in servers.** `tests/test_embed.py` serves model files (with ranges, for resuming) and
  Ollama's `/api/embed` from a local HTTP server.
- **Real files, made in the test.** Office documents, PDFs (`tests/pdfmaker.py` writes one by hand),
  git repositories with merge conflicts, files in legacy encodings.
- `tests/test_review.py` holds a test for each bug an independent review found.

## Measuring search quality

```bash
sfind models download                      # once
python eval/evaluate.py --show             # hit@1, hit@5 and MRR per mode and kind of query
python eval/evaluate.py --haystack 3000    # the same, with the documents hidden among 3,000 generated files
python eval/make_corpus.py /tmp/big        # a large messy folder to try sfind on by hand
```

`eval/data.py` is a small synthetic set in English and Chinese. If you change how search scores
passages, run it before and after, with the default model and an English one, and say what changed
in the pull request. Add queries to it when you find a case it misses (synthetic text only).

## Pull requests

- Add a test that fails without your change.
- Test data and evaluation data are synthetic: never real notes, names, mail or documents.
