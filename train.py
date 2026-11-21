import json, os, time, collections
import numpy as np

np.random.seed(0)
TOTAL_SECONDS = 300 * 60   # 5 轮合计最多 300 分钟
MIN_COUNT = 30             # 出现少于 30 次的字不参与训练
CTX, D, H = 6, 32, 128
EPOCHS, BATCH, LR = 5, 4096, 0.6
F = np.float32

raw = open("corpus.txt", encoding="utf-8").read().replace("\r", "")
counts = collections.Counter(raw)
keep = set(c for c, n in counts.items() if n >= MIN_COUNT)
text = "".join(c for c in raw if c in keep)
chars = sorted(keep)
V = len(chars)
ix = {c: i for i, c in enumerate(chars)}
data = np.array([ix[c] for c in text], dtype=np.int32)
N = len(data) - CTX
print(f"保留字符 {V} 个（原始 {len(counts)} 个），训练样本 {N}")

emb = (np.random.randn(V, D) * 0.1).astype(F)
w1 = (np.random.randn(CTX * D, H) / np.sqrt(CTX * D)).astype(F)
b1 = np.zeros(H, dtype=F)
w2 = (np.random.randn(H, V) / np.sqrt(H)).astype(F)
b2 = np.zeros(V, dtype=F)

offs = np.arange(CTX)
T0 = time.time()
stop = False

for epoch in range(EPOCHS):
    perm = np.random.permutation(N)
    total, seen = 0.0, 0
    epoch_start = time.time()
    for s in range(0, N, BATCH):
        if time.time() - T0 > TOTAL_SECONDS:
            stop = True
            break
        idx = perm[s:s + BATCH]
        B = len(idx)
        xb = data[idx[:, None] + offs]
        yb = data[idx + CTX]

        x = emb[xb].reshape(B, CTX * D)
        h = np.tanh(x @ w1 + b1)
        logits = h @ w2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        total += float(-np.log(p[np.arange(B), yb] + 1e-12).sum())
        seen += B

        dl = p
        dl[np.arange(B), yb] -= 1
        dl /= B
        dw2 = h.T @ dl
        db2 = dl.sum(axis=0)
        dz = (dl @ w2.T) * (1 - h * h)
        dw1 = x.T @ dz
        db1 = dz.sum(axis=0)
        dx = (dz @ w1.T).reshape(B, CTX, D)
        demb = np.zeros_like(emb)
        np.add.at(demb, xb, dx)

        emb -= LR * demb
        w1 -= LR * dw1
        b1 -= LR * db1
        w2 -= LR * dw2
        b2 -= LR * db2

    pct = 100.0 * seen / N
    print(f"epoch {epoch + 1}/{EPOCHS} loss {total / max(seen, 1):.4f} "
          f"完成 {pct:.1f}% 用时 {time.time() - epoch_start:.0f} 秒")
    if stop:
        print(f"已达到总时间上限 300 分钟，在第 {epoch + 1} 轮中途停止训练")
        break

os.makedirs("android/app/src/main/assets", exist_ok=True)

def flat(a):
    return [round(float(v), 4) for v in a.ravel()]

model = {
    "chars": "".join(chars),
    "ctx": CTX, "dim": D, "hidden": H,
    "emb": flat(emb), "w1": flat(w1), "b1": flat(b1),
    "w2": flat(w2), "b2": flat(b2),
}
with open("android/app/src/main/assets/model.json", "w", encoding="utf-8") as f:
    json.dump(model, f, ensure_ascii=False)

print(f"已导出模型，词表 {V} 个字，总用时 {(time.time() - T0) / 60:.1f} 分钟")
