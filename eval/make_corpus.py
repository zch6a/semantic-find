"""Write a large, messy synthetic folder for trying semantic-find at size: thousands of notes, Office
documents, PDFs and code files in English and Chinese, with duplicates, near-duplicates, empty and
damaged files, deep folders and clutter that should be skipped.

    python eval/make_corpus.py OUT_FOLDER [--files 3000] [--seed 1]

Everything is generated from templates and random choices; none of it is real data. A few "needle"
documents with known content are planted, so that searches on the big folder have known answers.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from pdfmaker import make_pdf  # noqa: E402

TOPICS_EN = {
    "meeting": ["The team agreed to {verb} the {thing} by {when}.", "{name} will follow up on the {thing}.", "Open question: who owns the {thing}?"],
    "recipe": ["Mix the {food} with {food2} and cook for {n} minutes.", "Season with salt and serve with {food2}.", "Keeps in the fridge for {n} days."],
    "travel": ["Flight to {city} leaves at {n}:00.", "Hotel near the station in {city}, {n} nights.", "Remember the adapter for {city}."],
    "finance": ["Paid {n}0 for the {thing} this month.", "The {thing} invoice is due on the {n}th.", "Budget review: cut spending on {thing}."],
    "health": ["Walked {n} kilometres today.", "Doctor's appointment about the {body} on {when}.", "Drink more water, sleep before eleven."],
    "code": ["The {thing} service times out under load.", "Refactor the {thing} module before release.", "Add tests for the {thing} parser."],
}
TOPICS_ZH = {
    "会议": ["大家同意在{when_zh}之前{verb_zh}{thing_zh}。", "{name_zh}负责跟进{thing_zh}。", "还没定下来的是：{thing_zh}由谁负责？"],
    "做饭": ["把{food_zh}和{food2_zh}一起炒{n}分钟。", "加盐调味，配{food2_zh}一起吃。", "放冰箱可以保存{n}天。"],
    "出行": ["去{city_zh}的航班{n}点起飞。", "酒店在{city_zh}火车站附近，住{n}晚。", "记得带{city_zh}能用的转换插头。"],
    "记账": ["这个月{thing_zh}花了{n}百块。", "{thing_zh}的账单{n}号到期。", "预算复盘：{thing_zh}的开销要减。"],
    "健康": ["今天走了{n}公里。", "{when_zh}去医院看{body_zh}。", "多喝水，十一点前睡觉。"],
}
WORDS = {
    "verb": ["finish", "review", "ship", "rewrite", "test", "document"], "thing": ["billing", "login page", "search", "report", "backup", "API", "dashboard"],
    "when": ["Friday", "next week", "the end of the month", "Monday", "the release"], "name": ["Ann", "Ben", "Chen", "Dana", "Eli"],
    "food": ["rice", "lentils", "chicken", "tofu", "potatoes"], "food2": ["onions", "garlic", "spinach", "carrots", "beans"],
    "city": ["Lisbon", "Osaka", "Berlin", "Toronto", "Nairobi"], "body": ["knee", "back", "eyes", "teeth"],
    "verb_zh": ["完成", "检查", "上线", "重写", "测试"], "thing_zh": ["账单系统", "登录页", "搜索", "报表", "备份", "接口", "看板"],
    "when_zh": ["周五", "下周", "月底", "周一", "发布"], "name_zh": ["小王", "小李", "老张", "小陈", "阿梅"],
    "food_zh": ["米饭", "豆腐", "鸡肉", "土豆", "白菜"], "food2_zh": ["洋葱", "大蒜", "菠菜", "胡萝卜", "青椒"],
    "city_zh": ["东京", "柏林", "成都", "西安", "大理"], "body_zh": ["膝盖", "腰", "眼睛", "牙"],
}
# Documents with known content, for searches with known answers on the big folder.
NEEDLES = {
    "archive/2023/house/insurance-claim.md": "# Water damage claim\n\nThe pipe under the kitchen sink burst in March. The insurer needs photos, "
                                             "the plumber's invoice and the claim form within 30 days.",
    "archive/2024/car/service.docx": "Annual car service: change the oil and the brake fluid, check the timing belt at 100,000 km.",
    "work/projects/search/design.pdf": "Design notes: the ranking combines keyword overlap with embedding similarity.",
    "笔记/生活/宠物.md": "# 猫\n\n猫粮每天两顿，每顿四十克。每年打一次疫苗，三月份带它去宠物医院。",
    "笔记/工作/周报模板.txt": "周报写三部分：本周完成的事、遇到的问题、下周计划。每周五下班前发给组长。",
}


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, values in WORDS.items():
        out = out.replace("{" + key + "}", rng.choice(values))
    return out.replace("{n}", str(rng.randint(2, 9)))


def paragraph(rng: random.Random, zh: bool) -> tuple[str, str]:
    topics = TOPICS_ZH if zh else TOPICS_EN
    topic = rng.choice(list(topics))
    lines = [fill(rng.choice(topics[topic]), rng) for _ in range(rng.randint(2, 6))]
    return topic, ("" if zh else " ").join(lines)


def docx(text: str) -> bytes:
    paras = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in text.split("\n"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{paras}</w:body></w:document>')
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--files", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--long", type=int, default=1, metavar="N", help="files N times longer (for many passages without many files)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    folders = ["notes", "notes/daily", "work/meetings", "work/projects/alpha", "work/projects/beta", "home", "recipes", "travel/2025",
               "笔记", "笔记/日记", "笔记/工作", "笔记/生活", "archive/2023", "archive/2024/misc", "src/app", "src/lib/util"]
    written = 0
    for i in range(args.files):
        zh = rng.random() < 0.4
        folder = rng.choice([f for f in folders if (f.startswith("笔记") or f.startswith("archive")) == zh or f.startswith("archive")])
        topic, text = paragraph(rng, zh)
        body = "\n\n".join([text] + [paragraph(rng, zh)[1] for _ in range(rng.randint(0, 8) * args.long + (args.long - 1) * 4)])
        kind = rng.random()
        path = root / folder / f"{topic}-{i:05d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind < 0.6:
            path.with_suffix(".md").write_text(f"# {topic} {i}\n\n{body}\n", encoding="utf-8")
        elif kind < 0.72:
            path.with_suffix(".txt").write_text(body, encoding="gb18030" if zh and rng.random() < 0.5 else "utf-8")
        elif kind < 0.8:
            path.with_suffix(".docx").write_bytes(docx(body))
        elif kind < 0.84 and not zh:
            path.with_suffix(".pdf").write_bytes(make_pdf([p[:80].replace("(", "").replace(")", "") for p in body.split("\n\n")[:3]]))
        elif kind < 0.92 and not zh:
            name = rng.choice(WORDS["thing"]).replace(" ", "_")
            (root / "src" / "app").mkdir(parents=True, exist_ok=True)
            (root / "src" / "app" / f"{name}_{i}.py").write_text(
                f'def handle_{name}(request):\n    """{text}"""\n    return process(request)\n', encoding="utf-8")
        else:
            path.with_suffix(".md").write_text(body, encoding="utf-8")
        written += 1
    # near-duplicates and exact duplicates
    for i in range(50):
        src = root / "notes" / f"copy-{i}.md"
        src.write_text("# Copied note\n\nThe cat flap needs a new magnet; the hardware shop on the corner has them.\n" + ("(edited)" if i % 2 else ""), encoding="utf-8")
    # what should be skipped, damaged or empty
    (root / "node_modules" / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "node_modules" / "pkg" / "README.md").write_text("insurance claim water damage pipe", encoding="utf-8")
    (root / ".obsidian").mkdir(exist_ok=True)
    (root / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    (root / "empty.md").write_text("", encoding="utf-8")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 broken")
    (root / "photo.jpg").write_bytes(bytes(rng.randrange(256) for _ in range(4000)))
    deep = root.joinpath(*[f"level{n}" for n in range(12)])
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "deep note.md").write_text("A note twelve folders down, about the garden shed roof.", encoding="utf-8")
    for rel, text in NEEDLES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".docx"):
            p.write_bytes(docx(text))
        elif rel.endswith(".pdf"):
            p.write_bytes(make_pdf([text]))
        else:
            p.write_text(text, encoding="utf-8")
    print(f"wrote {written + 50 + len(NEEDLES) + 5} files under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
