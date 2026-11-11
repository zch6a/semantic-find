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
- **关键词和意思一起用。** 每一段按意思（向量相似度）和它包含了查询里多少词（越少见的词分量越重）一起打分。像 `fetchWithRetry` 这样的确切名称和 grep 一样准，用大白话提的问题也能找到用词不同的答案。每条结果都会说明为什么匹配。[效果](#找得准不准)是实测的，不是估计的。
- **中文以及各种语言。** 默认模型支持 101 种语言；关键词搜索把中文、日文、韩文按单字和相邻两个字切分，所以 `sfind "数据库连接池怎么配置"` 不需要词典也能用。用一种语言搜索，常常也能找到另一种语言写的文件（用“what tires for snow”能找到“记得把车的轮胎换成雪地胎”），只是不如同一种语言可靠。
- **不只是代码，还有你的文档。** Markdown、纯文本、各种语言的代码、HTML、Word（.docx）、PowerPoint（.pptx）、Excel（.xlsx）、OpenDocument、EPUB、Jupyter 笔记本，以及 PDF（需 `pip install "semantic-find[pdf]"`）。旧编码的文件（GB18030、Big5、Shift-JIS、EUC-KR、Windows-1252）也能识别并读取。损坏的文件会被跳过，不会影响整个文件夹。
- **小而快。** 模型是静态向量模型（[Model2Vec](https://github.com/MinishLab/model2vec)）：不运行神经网络，所以不需要 PyTorch，也不需要显卡。sfind 本身只依赖 numpy 和 Hugging Face 的 `tokenizers`；`tokenizers` 会带上 `huggingface_hub` 及其依赖，一共大约 20 个小包。
- **说明有多确定。** 强匹配之后的弱匹配会被去掉。在有 500 个以上段落的文件夹里，如果没有任何强匹配（比如搜索文件夹里没有的东西），sfind 会明确说明，并把最接近的三段作为“可能不是你要找的”列出来，同时以退出码 1（“没找到”）结束，方便脚本判断；小文件夹里内容太少，无从判断哪段“突出”，所以只是列出最接近的段落。在非常大的文件夹里（几万个段落），许多“有点像”的段落中最好的那个可能碰巧显得突出，所以搜索不存在的东西时仍可能列出几段。包含查询里所有词的段落一定会显示。完全相同的段落只显示一次（显示为“the same text in 24 other files”）。
- **用法像 grep。** `-g "*.md"` 只搜部分文件，`-l` 只列文件名（最好的 `-k` 个，默认 10 个，以及它们的副本），`--json` 给脚本用，什么都没找到时退出码为 1，出错时为 2。

## 隐私

- **任何内容都不会离开你的电脑。** 索引是一个 SQLite 文件，放在你的用户数据目录里（`sfind status` 会显示位置），从不放进被搜索的文件夹。搜索不需要联网。
- **模型只下载一次，而且要你同意。** `sfind` 下载前会先问你（也可以直接运行 `sfind models download`）。内置模型的每个文件都在代码里固定了版本、大小和 SHA-256，所以镜像无法篡改。可以设置 `HF_ENDPOINT` 从镜像下载，例如 `HF_ENDPOINT=https://hf-mirror.com`。
- 在下载模型之前，`sfind` 只按关键词搜索，并会说明这一点。
- 使用 `--model ollama:<名称>` 时，文本会发给 Ollama，而且只发给本机上的 Ollama（不经过代理，也不使用云端模型）。
- 索引里包含你文件的文字内容，所以在 Linux 和 macOS 上只有你自己能读。`sfind forget 文件夹` 会删除它。

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

在 git 仓库里，会搜索 git 跟踪的文件，以及没被 git 忽略的未跟踪文件，但跳过依赖和构建目录（`node_modules`、`.venv`、`dist` 等）。在其他地方，这些目录和隐藏目录都会跳过。文件夹里的 `.sfindignore` 文件可以再列出要跳过的模式。同一个文件夹同时运行两个 `sfind` 也没问题：一个更新索引，另一个按现有索引搜索。

### 模型

| 模型 | 语言 | 下载大小 | |
|---|---|---|---|
| `potion-multilingual-128M`（默认） | 101 种，包括中文和英文 | 506 MB | 一个模型搞定任何语言的文档 |
| `potion-retrieval-32M` | 英文 | 125 MB | 专为搜索训练 |
| `potion-base-8M` | 英文 | 29 MB | 最小，适合先试试 |
| `ollama:<名称>` | 取决于模型 | 通过 Ollama | 例如 `ollama:bge-m3` 或 `ollama:qwen3-embedding`：更慢，但更准 |

用 `--model` 或环境变量 `SFIND_MODEL` 选择。放着 Model2Vec 模型的文件夹也可以直接用。

## 工作原理

1. **文件**：列出文件（在仓库里用 `git ls-files`），只读取新增或改动过的（按大小和修改时间判断）。
2. **段落**：按段落把文字切成约 700 字符、最多 1500 字符的段落；Markdown 标题会开始新的一段，并作为后面段落的上下文保留。
3. **关键词**：每段的词放进 SQLite FTS5 表；标识符还会拆成各个部分（`fetchWithRetry` → fetch、with、retry），中日韩文字按单字和相邻两个字切分。
4. **意思**：每段连同文件名和标题一起变成一个向量，也就是静态模型里各个词元向量的平均值，以 float16 保存。
5. **搜索**：每个候选段落的得分是它与查询的相似度（相对于最佳结果），加上 0.15 乘以它包含查询词的比例（越少见的词分量越重，在所有段落里都没有出现的词分量最重，所以“xylophone lessons”不会因为某段有“lessons”就算匹配）。同时搜索几个文件夹时，都以所有文件夹里的最佳匹配为基准打分。这是对归一化分数的凸组合，Bruch、Gai 和 Ingber 的研究发现它优于按名次融合（“An Analysis of Fusion Functions for Hybrid Retrieval”，ACM TOIS 2023）。最初用的是常见的倒数排名融合（RRF），但实测更差：一段只和查询共有四分之一的词的文字，会排在真正意思相符的那段前面。
6. **显示什么**：只靠意思找到的段落，相似度至少要达到最佳结果的一半。在有 500 个以上段落的文件夹里，只有“突出”的才算强匹配：比整个文件夹与查询的平均相似度高出至少三个标准差；否则算弱匹配，只有在没有任何强匹配时才显示。之所以标出来而不是直接去掉，是因为没有办法可靠地区分：在最多 64000 个段落上实测（`eval/standout.py`），最弱的那一成正确答案，和文件夹里根本没有的主题的最佳匹配看起来一模一样。得分不到最佳结果 60% 的会被去掉，每个文件显示最好的一段，完全相同的段落只显示一次。


## 找得准不准

用 `eval/evaluate.py` 在 `eval/data.py` 上实测：53 个中英文的合成笔记、文档和代码文件（其中一些故意和别的很像），130 个有标准答案的查询。查询的类别由评测程序根据索引自己判断：**原词**查询用的是答案里的原词，**部分相同**的查询和答案至少有一个相同的词（包括文件名），**无相同词**的查询和答案没有任何相同的词，**跨语言**查询用的是另一种语言。sfind 会标为“弱匹配”的结果算作没找到。“第 1”是正确文件排在第一的比例。

| 混合模式（默认），默认模型 | 原词 | 部分相同 | 无相同词 | 跨语言 | 全部，第 1 | 全部，前 5 |
|---|---|---|---|---|---|---|
| 用来选参数的 100 个查询 | 100% | 93% | 54% | 70% | 78% | 91% |
| 30 个留出查询（从未用来选任何参数） | 75% | 100% | （只有 1 个） | 40% | 87% | 93% |
| 这 100 个查询，文档混在 3000 个其他文件里 | 100% | 71% | 50% | 40% | 62% | 68% |

只用关键词搜索时，原词查询全对，“部分相同”的查询对 82%，其余都找不到；只按意思搜索时，100 个查询里 74% 第一个就是对的。用纯英文的 `potion-base-8M` 时，混合搜索第一名正确率为 50%，跨语言什么都找不到，这在意料之中。第三行更接近真实的文件夹：同样的文档混在 3000 个主题相近的生成笔记、Office 文件、PDF 和代码里。

实际使用中意味着：确切的名称和词和 grep 一样能找到；换了说法的问题，大约一半第一个就是对的，大多数情况下在前五个里；跨语言是静态模型的弱项。留出查询最后和答案共用的词比原本打算的多（30 个里有 20 个），所以它是更容易的测试，而不是更公平的测试；整个评测集也很小。这些数字只能当作基本的检验，而不是严格的基准测试。

## 状态

0.1，开发中。用合成的笔记和一个手工构造的小模型测试（测试不需要下载任何东西），并手动试过 Python 标准库（728 个文件、12254 个段落，在笔记本上 5 秒建完索引）、一个生成的 3060 个混杂文件的文件夹（`eval/make_corpus.py`；文件已在磁盘缓存里时 2.5 秒建完索引），以及一个 6060 个较长文件的文件夹：76915 个段落 49 秒建完索引，索引文件 195 MB，每次搜索约 2 秒（含启动）。在 Windows 上，刚写入的文件第一次读取会很慢，因为杀毒软件在扫描：这 3060 个文件要 35 秒。用多语言模型每次搜索约 2 秒，其中 1 秒是加载它 18 MB 的分词器；用英文模型不到 1 秒。

## 开发

```bash
uv venv && uv pip install -e ".[pdf]" pytest ruff
uv run pytest
```

参见 [CONTRIBUTING.md](CONTRIBUTING.md)（英文）和 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

[MIT](LICENSE)
