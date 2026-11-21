import json, collections, os

text = open("corpus.txt", encoding="utf-8").read().replace("\r", "")
chars = sorted(set(text))
ix = {c: i for i, c in enumerate(chars)}

uni = collections.Counter(ix[c] for c in text)
bi = collections.defaultdict(collections.Counter)
tri = collections.defaultdict(collections.Counter)
for i in range(1, len(text)):
    bi[text[i - 1]][ix[text[i]]] += 1
for i in range(2, len(text)):
    tri[text[i - 2:i]][ix[text[i]]] += 1

model = {
    "chars": chars,
    "uni": list(uni.items()),
    "bi": {k: list(v.items()) for k, v in bi.items()},
    "tri": {k: list(v.items()) for k, v in tri.items() if sum(v.values()) >= 2},
}

out_dir = "android/app/src/main/assets"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
    json.dump(model, f, ensure_ascii=False, separators=(",", ":"))

print(f"词表 {len(chars)} 个字，训练完成")
