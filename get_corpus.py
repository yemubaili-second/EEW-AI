import re, urllib.request

BOOKS = {
    # 中文（原有）
    24032: "儒林外史",
    26970: "彭公案",
    25559: "中国小说史略",
    # 英文小说（原有）
    1342: "Pride and Prejudice",
    11: "Alice's Adventures in Wonderland",
    # 数学（原有）
    68662: "Elements of Arithmetic",
    39088: "On the Study and Difficulties of Mathematics",
    33283: "Calculus Made Easy",
    41568: "An Introduction to Mathematics",
    39702: "The Philosophy of Mathematics",
    # 新增 10 本英文小说
    84: "Frankenstein",
    2701: "Moby Dick",
    345: "Dracula",
    98: "A Tale of Two Cities",
    174: "The Picture of Dorian Gray",
    43: "The Strange Case of Dr Jekyll and Mr Hyde",
    1661: "The Adventures of Sherlock Holmes",
    74: "The Adventures of Tom Sawyer",
    1184: "The Count of Monte Cristo",
    1727: "The Odyssey",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

def strip_gutenberg(raw):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    if m:
        raw = raw[m.end():]
    m = re.search(r"\*\*\* ?END OF.*?\*\*\*", raw)
    if m:
        raw = raw[:m.start()]
    return raw.strip()

parts = []
for book_id, name in BOOKS.items():
    url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")
        body = strip_gutenberg(raw)
        parts.append(body)
        print(f"OK  {name}: {len(body)} 字符")
    except Exception as e:
        print(f"跳过 {name}: {e}")

text = "\n\n".join(parts)
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"语料合计 {len(text)} 字符，共 {len(parts)} 本")
