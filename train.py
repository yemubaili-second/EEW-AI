import json, os
import numpy as np

np.random.seed(0)
text = open("corpus.txt", encoding="utf-8").read().replace("\r", "")
chars = sorted(set(text))
V = len(chars)
ix = {c: i for i, c in enumerate(chars)}
data = np.array([ix[c] for c in text], dtype=np.int64)

CTX, D, H = 6, 32, 128
EPOCHS, BATCH, LR = 5, 512, 0.3

X = np.stack([data[i:i + CTX] for i in range(len(data) - CTX)])
Y = data[CTX:]
N = len(Y)

emb = np.random.randn(V, D) * 0.1
w1 = np.random.randn(CTX * D, H) / np.sqrt(CTX * D)
b1 = np.zeros(H)
w2 = np.random.randn(H, V) / np.sqrt(H)
b2 = np.zeros(V)

for epoch in range(EPOCHS):
    perm = np.random.permutation(N)
    total = 0.0
    for s in range(0, N, BATCH):
        idx = perm[s:s + BATCH]
        xb, yb = X[idx], Y[idx]
        B = len(idx)

        x = emb[xb].reshape(B, CTX * D)
        h = np.tanh(x @ w1 + b1)
        logits = h @ w2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        total += -np.log(p[np.arange(B), yb] + 1e-12).sum()

        dl = p.copy()
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

    print(f"epoch {epoch + 1}/{EPOCHS} loss {total / N:.4f}")

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

print(f"已导出模型，词表 {V} 个字")
