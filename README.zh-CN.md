<div align="center">

# semantic-find

**按意思搜索你的文件，完全离线。** `sfind` 是“按意思的 grep”：用自己的话说出要找的东西（中文、英文或其他 100 种语言都可以），它就能在你的笔记、文档和代码里找到那一段，即使用词完全不同。

<a href="https://github.com/zch6a/semantic-find/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zch6a/semantic-find/actions/workflows/ci.yml/badge.svg"></a>
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
<img alt="离线" src="https://img.shields.io/badge/your%20files-stay%20local-2ea44f">
<a href="LICENSE"><img alt="MIT 许可证" src="https://img.shields.io/badge/license-MIT-blue"></a>

[English](README.md) · **简体中文**

</div>

---

```console
$ sfind "how long does getting a new travel document take" ~/notes
notes/travel.md:1
    Travel — Passport renewal: book an appointment at the post office, bring the old passport
    and two photos. It takes about six weeks.
    (words: travel · meaning 0.64 · 1 more in this file)
```

查询里没有“passport”，文件里也没有“travel document”：这是按意思匹配到的。

## 它能做什么

- **不用配置。** `sfind 查询 [文件夹]` 第一次会先建索引再搜索；之后只重新读取改动过的文件，所以马上就能出结果。
- **关键词和意思一起用。** 每一段都会分别按关键词（BM25）和意思（向量相似度）排序，再把两个排序融合（倒数排名融合，RRF）。像 `fetchWithRetry` 这样的确切名称和 grep 一样准，用大白话提的问题也能找到用词不同的答案。每条结果都会说明为什么匹配。
- **中文以及各种语言。** 默认模型支持 101 种语言；关键词搜索把中文、日文、韩文按相邻两个字切分，所以 `sfind "数据库连接池怎么配置"` 不需要词典也能用。用一种语言搜索，常常也能找到另一种语言写的文件（用“what tires for snow”能找到“记得把车的轮胎换成雪地胎”），只是不如同一种语言可靠。
- **不只是代码，还有你的文档。** Markdown、纯文本、各种语言的代码、HTML、Word（.docx）、PowerPoint（.pptx）、Excel（.xlsx）、OpenDocument、EPUB、Jupyter 笔记本，以及 PDF（需 `pip install "semantic-find[pdf]"`）。GB18030 和 UTF-16 编码的文件也能读。
- **小而快。** 模型是静态向量模型（[Model2Vec](https://github.com/MinishLab/model2vec)）：不运行神经网络，所以不需要 PyTorch，也不需要显卡。依赖只有 numpy 和 Hugging Face 的 `tokenizers`。
- **用法像 grep。** `-g "*.md"` 只搜部分文件，`-l` 只列文件名，`--json` 给脚本用，什么都没找到时退出码为 1。

## 隐私

- **任何内容都不会离开你的电脑。** 索引是一个 SQLite 文件，放在你的用户数据目录里（`sfind status` 会显示位置），从不放进被搜索的文件夹。搜索不需要联网。
- **模型只下载一次，而且要你同意。** `sfind` 下载前会先问你（也可以直接运行 `sfind models download`），并按 Hugging Face 公布的大小和 SHA-256 校验。可以设置 `HF_ENDPOINT` 使用镜像，例如 `HF_ENDPOINT=https://hf-mirror.com`。
- 在下载模型之前，`sfind` 只按关键词搜索，并会说明这一点。
- 使用 `--model ollama:<名称>` 时，文本会发给 Ollama，而且只发给本机上的 Ollama（不经过代理，也不使用云端模型）。
- 索引里包含你文件的文字内容。`sfind forget 文件夹` 会删除它。

## 安装

```bash
pip install "semantic-find @ git+https://github.com/zch6a/semantic-find"
```

或者用 `uv tool install git+https://github.com/zch6a/semantic-find`（或 `pipx install ...`），只安装 `sfind` 命令、不影响其他 Python 项目。要搜索 PDF 请加上 `[pdf]`。

## 使用

```bash
sfind "护照续签" ~/Documents                 # 搜索一个文件夹（第一次会先建索引）
sfind "数据库连接池怎么配置" .                 # 中文和英文一样好用
sfind "retry with backoff" src -g "*.py"    # 只搜部分文件
sfind "2026 年预算" ~/Documents -k 20 -a      # 20 条结果，列出每一个匹配的段落
sfind "发票" ~/Documents -l                  # 只列路径，便于接管道
sfind "发票" ~/Documents --json              # 给脚本用
sfind "fetchWithRetry" --mode keyword        # 只按确切名称
sfind index ~/notes                          # 只建索引，不搜索
sfind status ~/notes                         # 索引了什么、索引在哪里
sfind forget ~/notes                         # 删除索引
sfind models                                 # 有哪些模型、哪些已下载
```

在 git 仓库里，git 忽略的文件不会被搜索。在其他地方，版本控制、依赖和构建目录（`node_modules`、`.venv`、`dist` 等）以及隐藏目录都会跳过。文件夹里的 `.sfindignore` 文件可以再列出要跳过的模式。

### 模型

| 模型 | 语言 | 下载大小 | |
|---|---|---|---|
| `potion-multilingual-128M`（默认） | 101 种，包括中文和英文 | 507 MB | 一个模型搞定任何语言的文档 |
| `potion-retrieval-32M` | 英文 | 125 MB | 专为搜索训练：纯英文文件效果更好 |
| `potion-base-8M` | 英文 | 30 MB | 最小，适合先试试 |
| `ollama:<名称>` | 取决于模型 | 通过 Ollama | 例如 `ollama:bge-m3` 或 `ollama:qwen3-embedding`：更慢，但更准 |

用 `--model` 或环境变量 `SFIND_MODEL` 选择。放着 Model2Vec 模型的文件夹也可以直接用。

## 工作原理

1. **文件**：列出文件（在仓库里用 `git ls-files`），只读取新增或改动过的（按大小和修改时间判断）。
2. **段落**：按段落把文字切成约 700 字符、最多 1500 字符的段落；Markdown 标题会开始新的一段，并作为后面段落的上下文保留。
3. **关键词**：每段的词放进 SQLite FTS5 表；标识符还会拆成各个部分（`fetchWithRetry` → fetch、with、retry），中日韩文字按相邻两个字切分。
4. **意思**：每段连同文件名和标题一起变成一个向量，也就是静态模型里各个词元向量的平均值，以 float16 保存。
5. **搜索**：查询按两种方式分别排序。只靠意思找到的段落，相似度至少要达到最佳结果的一半，所以搜索不存在的东西时会什么也找不到，而不是返回一堆噪音。两个排序融合后，每个文件显示最好的一段。

## 状态

0.1，开发中。用合成的笔记和一个手工构造的小模型测试（测试不需要下载任何东西），并在 Python 标准库上手动试过（728 个文件、12254 个段落，在笔记本上 17 秒建完索引，每次搜索约 1 秒）。

## 开发

```bash
uv venv && uv pip install -e ".[pdf]" pytest ruff
uv run pytest
```

## 许可证

[MIT](LICENSE)
