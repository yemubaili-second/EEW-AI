import re, urllib.request

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

HEADERS = {"User-Agent": "Mozilla/5.0"}

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

def fetch(book_id):
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
    ]
    last_error = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            raw_bytes = urllib.request.urlopen(req, timeout=60).read()
            return strip_gutenberg(decode(raw_bytes))
        except Exception as e:
            last_error = e
    raise last_error

parts = []
for book_id, name in BOOKS.items():
    try:
        body = fetch(book_id)
        parts.append(body)
        print(f"OK  {name}: {len(body)} 字符")
    except Exception as e:
        print(f"跳过 {name}: {e}")

text = "\n\n".join(parts)
print(f"语料合计 {len(text)} 字符，共 {len(parts)} 本")

if len(text) < 1_000_000:
    raise SystemExit("语料太少，下载失败，已停止。请查看上面的'跳过'行。")

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(text)
