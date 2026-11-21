import urllib.request

URL = "https://www.gutenberg.org/ebooks/24032.txt.utf-8"

raw = urllib.request.urlopen(URL).read().decode("utf-8-sig")

# 去掉 Project Gutenberg 开头和结尾的版权声明
start = raw.find("*** START OF")
end = raw.find("*** END OF")
if start != -1:
    body = raw[raw.find("\n", start) + 1:end] if end != -1 else raw[raw.find("\n", start) + 1:]
else:
    body = raw

body = body.strip()

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(body)

print(f"语料已保存，共 {len(body)} 个字符")
