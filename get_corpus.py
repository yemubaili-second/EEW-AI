import bz2, codecs, html, os, re, time, urllib.request

BOOKS = {
    # 中文
    24032: "儒林外史",
    26970: "彭公案",
    25559: "中国小说史略",
    23962: "西游记",
    23839: "论语",
    24264: "红楼梦",
    # 英文小说
    1342: "Pride and Prejudice",
    11: "Alice's Adventures in Wonderland",
    84: "Frankenstein",
    2701: "Moby Dick",
    345: "Dracula",
    98: "A Tale of Two Cities",
    174: "The Picture of Dorian Gray",
    43: "Dr Jekyll and Mr Hyde",
    1661: "The Adventures of Sherlock Holmes",
    74: "The Adventures of Tom Sawyer",
    1184: "The Count of Monte Cristo",
    1727: "The Odyssey",
    1400: "Great Expectations",
    1260: "Jane Eyre",
    158: "Emma",
    161: "Sense and Sensibility",
    # 常用语
    48441: "Putnam's Phrase Book",
    27830: "Conversation: What to say and how to say it",
    39281: "A Dictionary of English Proverbs",
    27889: "Familiar Quotations",
    77369: "Wise-crack Dictionary",
    # 数学
    68662: "Elements of Arithmetic",
    39088: "On the Study and Difficulties of Mathematics",
    33283: "Calculus Made Easy",
    41568: "An Introduction to Mathematics",
    39702: "The Philosophy of Mathematics",
    38769: "A Course of Pure Mathematics",
}

# 维基百科（编号 9001、9002 为本地编号，不与 Gutenberg 冲突）
WIKIS = [
    (9001, "中文维基百科", "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2", 20_000_000),
    (9002, "英文维基百科", "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2", 30_000_000),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EEW-AI-corpus/1.0; contact: github)"}
MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
]

DOWNLOAD_BUDGET = 30 * 60   # Gutenberg 下载总时间最多 30 分钟
WIKI_BUDGET = 60 * 60       # 维基百科下载总时间最多 60 分钟
BOOK_DIR = "books"

os.makedirs(BOOK_DIR, exist_ok=True)

# ---------- Gutenberg 书籍 ----------

def decode(raw_bytes):
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")

def strip_gutenberg(raw):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    if m:
        raw = raw[m.end():]
    m = re.search(r"\*\*\* ?END OF.*?\*\*\*", raw)
    if m:
        raw = raw[:m.start()]
    return raw.strip()

def fetch_book(book_id):
    last_error = None
    for template in MIRRORS:
        url = template.format(id=book_id)
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                raw_bytes = urllib.request.urlopen(req, timeout=30).read()
                return strip_gutenberg(decode(raw_bytes))
            except Exception as e:
                last_error = e
                time.sleep(3)
    raise last_error

def download_books(T0):
    for book_id, name in BOOKS.items():
        path = os.path.join(BOOK_DIR, f"{book_id}.txt")
        if os.path.exists(path):
            print(f"缓存 {name}", flush=True)
            continue
        if time.time() - T0 > DOWNLOAD_BUDGET:
            print(f"下载时间已到 30 分钟，跳过剩余书籍：{name} 及之后", flush=True)
            continue
        try:
            body = fetch_book(book_id)
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            print(f"OK  {name}: {len(body)} 字符", flush=True)
        except Exception as e:
            print(f"跳过 {name}: {e}", flush=True)
        time.sleep(1)

# ---------- 维基百科：边下边解压，够字数就停 ----------

PAGE_RE = re.compile(r"<page>.*?</page>", re.S)

def clean_wiki(t):
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*?/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    prev = None
    while prev != t:                      # 反复去掉模板（可能嵌套）
        prev = t
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)                      # 表格
    t = re.sub(r"\[\[(?:File|Image|Category|文件|分类)[:：][^\[\]]*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]", r"\1", t)         # [[a|b]] -> b
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)                 # 外链保留文字
    t = re.sub(r"\[https?://\S+\]", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)          # 标题
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def pages_to_texts(xml_pages):
    for p in xml_pages:
        ns = re.search(r"<ns>(\d+)</ns>", p)
        if not ns or ns.group(1) != "0":
            continue                      # 只要正文条目
        if "<redirect" in p:
            continue
        tm = re.search(r"<text[^>]*>(.*?)</text>", p, re.S)
        if not tm:
            continue
        text = clean_wiki(html.unescape(tm.group(1)))
        if len(text) >= 200:              # 跳过过短的条目
            yield text

def stream_wiki(url, target):
    resp = urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120)
    dec = bz2.BZ2Decompressor()
    utf = codecs.getincrementaldecoder("utf-8")(errors="ignore")   # 注意：这里不加 ()
    buf = ""
    parts, total = [], 0
    while total < target and not dec.eof:
        chunk = resp.read(1 << 20)
        if not chunk:
            break
        buf += utf.decode(dec.decompress(chunk))
        last = 0
        for m in PAGE_RE.finditer(buf):
            last = m.end()
            for text in pages_to_texts([m.group(0)]):
                parts.append(text)
                total += len(text)
                if total >= target:
                    break
            if total >= target:
                break
        buf = buf[last:]
    text = "\n\n".join(parts)
    return text[:target]

def download_wikis(T0):
    for wid, name, url, target in WIKIS:
        path = os.path.join(BOOK_DIR, f"{wid}.txt")
        if os.path.exists(path):
            print(f"缓存 {name}", flush=True)
            continue
        if time.time() - T0 > WIKI_BUDGET:
            print(f"维基下载时间已到 60 分钟，跳过 {name}", flush=True)
            continue
        print(f"开始下载 {name}，目标 {target} 字符", flush=True)
        for attempt in range(3):
            try:
                body = stream_wiki(url, target)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(body)
                print(f"OK  {name}: {len(body)} 字符", flush=True)
                break
            except Exception as e:
                print(f"{name} 第 {attempt + 1} 次失败: {e}", flush=True)
                time.sleep(10)

# ---------- 主流程 ----------

T0 = time.time()
download_books(T0)
download_wikis(time.time())

total_chars = 0
count = 0
for fn in os.listdir(BOOK_DIR):
    if fn.endswith(".txt"):
        count += 1
        total_chars += os.path.getsize(os.path.join(BOOK_DIR, fn))
print(f"书籍目录共 {count} 个文件，约 {total_chars / 1_000_000:.1f} MB，用时 {(time.time() - T0) / 60:.1f} 分钟", flush=True)

if count < 5:
    raise SystemExit("书籍太少，下载失败，已停止。请查看上面的'跳过'行。")

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("")
