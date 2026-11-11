"""A small evaluation set: synthetic notes, documents and code in English and Chinese, and queries whose
right answer is known. Every text here was written for this file; none is anyone's real data.

Queries come in three kinds:
- "words": the query shares its key words with the answer (what grep is good at);
- "meaning": the query says the same thing in other words;
- "cross": the query is in one language and the answer in the other.
Several documents are near-misses for others (two about passports, two about databases), so a search
that only matches the topic is not enough.
"""

DOCS = {
    "travel/passport-renewal.md": "# Renewing a passport\n\nBook an appointment at the passport office, bring the old passport, two recent photos and the fee. "
                                  "A new passport arrives in about six weeks; an urgent one can be had in a week for a higher fee.",
    "travel/lost-passport.md": "# If a passport is lost abroad\n\nReport it to the local police, then go to the embassy with the police report. "
                               "The embassy issues an emergency travel document valid for one journey home.",
    "travel/visa-japan.md": "# Japan visa\n\nCitizens of many countries need no visa for stays under 90 days. Others apply at the consulate "
                            "with an itinerary, a bank statement and a flight booking.",
    "travel/packing.md": "# Packing list\n\nCharger, adapter plug, toothbrush, medicine, a printed copy of the hotel booking, and a light rain jacket.",
    "home/boiler.md": "# Boiler pressure\n\nIf the pressure gauge reads below 1 bar, open the filling loop slowly until it reaches 1.5 bar, "
                      "then close it. Bleed the radiators if they stay cold at the top.",
    "home/plants.md": "# Houseplants\n\nWater the fern twice a week and keep it out of direct sun. The cactus needs water only once a month in winter.",
    "home/recycling.txt": "Glass goes in the green bin, paper and cardboard in the blue bin. Batteries are taken back at the supermarket.",
    "cooking/bread.md": "# Sourdough bread\n\nMix flour, water, salt and starter. Let the dough rise overnight, shape it, and bake at 230 °C for 40 minutes.",
    "cooking/rice.md": "# Rice\n\nRinse the rice until the water runs clear. One cup of rice to one and a half cups of water, simmer covered for 15 minutes.",
    "money/tax-deadline.md": "# Tax return\n\nThe return for last year must be filed by 31 July. Keep receipts for work equipment: they are deductible.",
    "money/budget.md": "# Monthly budget\n\nRent 1,200, groceries 400, transport 120, savings 300. Review the subscriptions every quarter.",
    "health/sleep.md": "# Sleeping better\n\nNo screens an hour before bed, a cool dark room, and the same wake-up time every day, weekends too.",
    "health/back-pain.md": "# Back pain\n\nStretch the hamstrings daily, take a short walk every hour at the desk, and raise the monitor to eye level.",
    "work/meeting-notes.md": "# Planning meeting\n\nDecided: the release moves to March. Anna owns the migration plan; Ben checks the licences.",
    "work/onboarding.md": "# First week\n\nAsk IT for a laptop and an account on day one. Read the style guide and pair with a colleague on a small fix.",
    "code/retry.py": "def fetch_with_retry(url, attempts=5):\n    # wait longer after each failure: 1, 2, 4, 8 seconds\n    for n in range(attempts):\n"
                     "        try:\n            return http_get(url)\n        except ConnectionError:\n            time.sleep(2 ** n)\n    raise RuntimeError('gave up')",
    "code/cache.py": "class LRUCache:\n    \"\"\"Keeps the most recently used items and evicts the oldest when full.\"\"\"\n    def __init__(self, capacity):\n"
                     "        self.items = OrderedDict()\n        self.capacity = capacity",
    "code/config_loader.py": "def load_settings(path):\n    \"\"\"Read the settings from a TOML file, falling back to defaults for missing keys.\"\"\"\n"
                             "    with open(path, 'rb') as f:\n        return {**DEFAULTS, **tomllib.load(f)}",
    "code/email_check.py": "def is_valid_email(address):\n    \"\"\"True if the address has one @, a name before it and a domain with a dot after it.\"\"\"\n"
                           "    name, _, domain = address.partition('@')\n    return bool(name) and '.' in domain",
    # 中文
    "笔记/数据库连接池.md": "# 连接池配置\n\n应用启动时创建连接池，最大连接数设为 20，最小空闲连接 5 个，空闲超过 300 秒的连接会被回收。",
    "笔记/数据库备份.md": "# 数据库备份\n\n每天凌晨三点做一次全量备份，保留七天；每小时做一次增量备份。恢复前先在测试环境演练。",
    "笔记/护照换发.md": "# 护照换发\n\n护照有效期不足六个月时，要先到出入境管理局换发新护照，带身份证和旧护照，大约七个工作日可以拿到。",
    "笔记/租房.md": "# 租房注意事项\n\n签合同前看清押金怎么退、水电费谁付、能不能提前退租。入住时拍照记录屋里的家具和损坏。",
    "笔记/做饭.md": "# 红烧肉\n\n五花肉切块焯水，加冰糖炒出糖色，再加酱油、料酒和热水，小火炖一个小时。",
    "笔记/轮胎.txt": "冬天到了，记得把车的轮胎换成雪地胎。胎压每个月检查一次，低于 2.2 要充气。",
    "笔记/睡眠.md": "# 改善睡眠\n\n睡前一小时不看手机，卧室保持凉爽和黑暗，每天同一时间起床，周末也一样。",
    "笔记/报销.md": "# 报销流程\n\n出差回来一周内提交报销单，附上发票原件和行程单，经理审批后财务月底打款。",
    "笔记/学英语.md": "# 学英语\n\n每天背二十个单词，听一集播客，周末找人练口语。不要只看不说。",
    "笔记/会议纪要.md": "# 周会纪要\n\n上线时间推迟到下个月十五号。小王负责测试，小李跟进客户反馈。",
}

# (query, the document that answers it, kind)
QUERIES = [
    ("passport renewal photos fee", "travel/passport-renewal.md", "words"),
    ("how long until I get a new passport", "travel/passport-renewal.md", "meaning"),
    ("lost my passport while travelling", "travel/lost-passport.md", "meaning"),
    ("emergency travel document embassy", "travel/lost-passport.md", "words"),
    ("do I need a visa for a short trip to Japan", "travel/visa-japan.md", "meaning"),
    ("what to bring on holiday", "travel/packing.md", "meaning"),
    ("heating system pressure too low", "home/boiler.md", "meaning"),
    ("radiators cold at the top", "home/boiler.md", "words"),
    ("how often should I water my plants", "home/plants.md", "meaning"),
    ("where do old batteries go", "home/recycling.txt", "meaning"),
    ("sourdough starter overnight", "cooking/bread.md", "words"),
    ("how to bake a loaf", "cooking/bread.md", "meaning"),
    ("water ratio for cooking rice", "cooking/rice.md", "meaning"),
    ("when is the tax return due", "money/tax-deadline.md", "meaning"),
    ("how much do I spend on rent and food", "money/budget.md", "meaning"),
    ("cannot fall asleep at night", "health/sleep.md", "meaning"),
    ("my back hurts from sitting at the computer", "health/back-pain.md", "meaning"),
    ("who is responsible for the migration", "work/meeting-notes.md", "meaning"),
    ("what to do on the first day at a new job", "work/onboarding.md", "meaning"),
    ("exponential backoff", "code/retry.py", "meaning"),
    ("fetch_with_retry", "code/retry.py", "words"),
    ("evict least recently used entries", "code/cache.py", "meaning"),
    ("read configuration file with defaults", "code/config_loader.py", "meaning"),
    ("check that an email address is well formed", "code/email_check.py", "meaning"),
    ("数据库连接池怎么配置", "笔记/数据库连接池.md", "words"),
    ("最多能开多少个数据库连接", "笔记/数据库连接池.md", "meaning"),
    ("数据多久备份一次", "笔记/数据库备份.md", "meaning"),
    ("护照快过期了怎么办", "笔记/护照换发.md", "meaning"),
    ("租房子要注意什么", "笔记/租房.md", "meaning"),
    ("押金能不能退", "笔记/租房.md", "words"),
    ("红烧肉怎么做", "笔记/做饭.md", "words"),
    ("车胎气压多少合适", "笔记/轮胎.txt", "meaning"),
    ("晚上睡不着", "笔记/睡眠.md", "meaning"),
    ("出差的钱怎么报", "笔记/报销.md", "meaning"),
    ("怎样提高英语水平", "笔记/学英语.md", "meaning"),
    ("上线日期改到什么时候", "笔记/会议纪要.md", "meaning"),
    # across languages
    ("database connection pool settings", "笔记/数据库连接池.md", "cross"),
    ("how often to back up the database", "笔记/数据库备份.md", "cross"),
    ("winter tyres for the car", "笔记/轮胎.txt", "cross"),
    ("braised pork belly recipe", "笔记/做饭.md", "cross"),
    ("expense claim after a business trip", "笔记/报销.md", "cross"),
    ("护照丢了", "travel/lost-passport.md", "cross"),
    ("锅炉压力低怎么办", "home/boiler.md", "cross"),
    ("怎么做面包", "cooking/bread.md", "cross"),
    ("重试时等待时间加倍", "code/retry.py", "cross"),
]
