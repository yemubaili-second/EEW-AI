import json, os, time, glob, collections
import numpy as np

RUN_SECONDS = 300 * 60   # 每次运行最多训练 300 分钟
MIN_COUNT = 1            # 出现少于 1 次的字符不纳入字表（即全部保留）
CTX, D, H = 6, 32, 128
BATCH, LR = 4096, 0.6
SAVE_EVERY = 300         # 每 300 批保存一次断点
STAGE_EPOCHS = 5         # 每组新书训练的轮数
REPLAY_RATIO = 0.3       # 训练新书时混入旧书内容的比例
CHARSET_VERSION = 5      # 字表规则版本：版本不一致时全部参数清空，从零训练
BOOK_DIR = "books"
CKPT_DIR = "ckpt"
CKPT = os.path.join(CKPT_DIR, "state.npz")
MODEL_OUT = "android/app/src/main/assets/model.json"
PARAMS = ["emb", "w1", "b1", "w2", "b2"]
F = np.float32

def load_books():
    books = {}
    for path in glob.glob(os.path.join(BOOK_DIR, "*.txt")):
        bid = int(os.path.splitext(os.path.basename(path))[0])
        with open(path, encoding="utf-8") as f:
            books[bid] = f.read().replace("\r", "")
    return books

def fresh_state():
    rng = np.random.default_rng(0)
    return {
        "emb": np.zeros((0, D), F),
        "w1": (rng.standard_normal((CTX * D, H)) / np.sqrt(CTX * D)).astype(F),
        "b1": np.zeros(H, F),
        "w2": np.zeros((H, 0), F),
        "b2": np.zeros(0, F),
        "chars": "", "trained": [], "stage": [],
        "stage_epoch": 0, "stage_batch": 0, "runs": 0,
        "ver": CHARSET_VERSION,
    }

def load_state():
    if not os.path.exists(CKPT):
        return fresh_state()
    z = np.load(CKPT)
    st = {k: z[k].astype(F) for k in PARAMS}
    st.update({
        "chars": str(z["chars"]),
        "trained": [int(x) for x in z["trained"]],
        "stage": [int(x) for x in z["stage"]],
        "stage_epoch": int(z["stage_epoch"]),
        "stage_batch": int(z["stage_batch"]),
        "runs": int(z["runs"]),
        "ver": int(z["ver"]) if "ver" in z.files else 1,
    })
    return st

def save_state(st):
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = os.path.join(CKPT_DIR, "tmp.npz")
    np.savez(tmp,
             emb=st["emb"], w1=st["w1"], b1=st["b1"], w2=st["w2"], b2=st["b2"],
             chars=np.array(st["chars"]),
             trained=np.array(st["trained"], dtype=np.int64),
             stage=np.array(st["stage"], dtype=np.int64),
             stage_epoch=np.array(st["stage_epoch"]),
             stage_batch=np.array(st["stage_batch"]),
             runs=np.array(st["runs"]),
             ver=np.array(st["ver"]))
    os.replace(tmp, CKPT)

def build_vocab(books, old_chars):
    counts = collections.Counter()
    for t in books.values():
        counts.update(t)
    old_set = set(old_chars)
    new = sorted(c for c, n in counts.items()
                 if n >= MIN_COUNT and c not in old_set)
    return old_chars + "".join(new), new

def expand(st, n, rng):
    # 新字符追加到字表末尾，已有字符的参数保持不变
    if n == 0:
        return
    st["emb"] = np.vstack([st["emb"], (rng.standard_normal((n, D)) * 0.1).astype(F)])
    st["w2"] = np.hstack([st["w2"], (rng.standard_normal((H, n)) / np.sqrt(H)).astype(F)])
    st["b2"] = np.concatenate([st["b2"], np.zeros(n, dtype=F)])

def encode(text, ix):
    return np.fromiter((ix[c] for c in text if c in ix), dtype=np.int32)

def make_stage(st, books, ix):
    trained = set(st["trained"])
    if st["stage"] and all(b in books for b in st["stage"]):
        stage_ids = list(st["stage"])
    else:
        stage_ids = [b for b in sorted(books) if b not in trained]
        if not stage_ids:
            return None
        st["stage"], st["stage_epoch"], st["stage_batch"] = stage_ids, 0, 0
        save_state(st)

    old_ids = [b for b in sorted(books) if b in trained]
    enc_stage = np.concatenate([encode(books[b], ix) for b in stage_ids])
    if old_ids:
        enc_old = np.concatenate([encode(books[b], ix) for b in old_ids])
    else:
        enc_old = np.zeros(0, dtype=np.int32)
    data = np.concatenate([enc_stage, enc_old])

    n_stage = len(enc_stage) - CTX
    starts = [np.arange(0, n_stage)]
    if len(enc_old) > CTX + 1:
        rng = np.random.default_rng(sum(stage_ids) % 100000 + len(old_ids))
        m = int(n_stage * REPLAY_RATIO)
        starts.append(rng.integers(len(enc_stage), len(data) - CTX, size=m))
    samples = np.concatenate(starts)
    return data, samples, stage_ids

def train_stage(st, data, samples, T0):
    offs = np.arange(CTX)
    nb = (len(samples) + BATCH - 1) // BATCH
    while st["stage_epoch"] < STAGE_EPOCHS:
        perm = np.random.default_rng(1000 + st["stage_epoch"]).permutation(len(samples))
        total, seen = 0.0, 0
        t_ep = time.time()
        for bi in range(st["stage_batch"], nb):
            if time.time() - T0 > RUN_SECONDS:
                st["stage_batch"] = bi
                save_state(st)
                return "timeout"
            idx = samples[perm[bi * BATCH:(bi + 1) * BATCH]]
            B = len(idx)
            xb = data[idx[:, None] + offs]
            yb = data[idx + CTX]

            emb, w1, b1, w2, b2 = st["emb"], st["w1"], st["b1"], st["w2"], st["b2"]
            x = emb[xb].reshape(B, CTX * D)
            h = np.tanh(x @ w1 + b1)
            logits = h @ w2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            total += float(-np.log(p[np.arange(B), yb] + 1e-12).sum())
            seen += B

            p[np.arange(B), yb] -= 1
            p /= B
            dw2 = h.T @ p
            db2 = p.sum(axis=0)
            dz = (p @ w2.T) * (1 - h * h)
            dw1 = x.T @ dz
            db1 = dz.sum(axis=0)
            dx = (dz @ w1.T).reshape(B, CTX, D)
            demb = np.zeros_like(emb)
            np.add.at(demb, xb, dx)

            st["emb"] -= LR * demb
            st["w1"] -= LR * dw1
            st["b1"] -= LR * db1
            st["w2"] -= LR * dw2
            st["b2"] -= LR * db2

            if (bi + 1) % SAVE_EVERY == 0:
                st["stage_batch"] = bi + 1
                save_state(st)

        print(f"第 {st['stage_epoch'] + 1}/{STAGE_EPOCHS} 轮 loss {total / max(seen, 1):.4f} "
              f"用时 {time.time() - t_ep:.0f} 秒", flush=True)
        st["stage_epoch"] += 1
        st["stage_batch"] = 0
        save_state(st)
    return "done"

def export(st):
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    def flat(a):
        return [round(float(v), 4) for v in a.ravel()]
    model = {
        "chars": st["chars"], "ctx": CTX, "dim": D, "hidden": H,
        "emb": flat(st["emb"]), "w1": flat(st["w1"]), "b1": flat(st["b1"]),
        "w2": flat(st["w2"]), "b2": flat(st["b2"]),
    }
    with open(MODEL_OUT, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)

os.makedirs(CKPT_DIR, exist_ok=True)
books = load_books()
if not books:
    raise SystemExit("books 目录里没有书，请先运行 get_corpus.py")

st = load_state()
if st["ver"] != CHARSET_VERSION:
    print(f"字表版本 {st['ver']} -> {CHARSET_VERSION}，清空全部参数和记录，从零开始训练", flush=True)
    st = fresh_state()

chars, new_chars = build_vocab(books, st["chars"])
expand(st, len(new_chars), np.random.default_rng(len(chars)))
st["chars"] = chars
st["runs"] += 1
ix = {c: i for i, c in enumerate(chars)}
print(f"书籍 {len(books)} 本，字符 {len(chars)} 个（本次新增 {len(new_chars)}），"
      f"已训练 {len(st['trained'])} 本，第 {st['runs']} 次运行", flush=True)

T0 = time.time()
status = "all_done"
while True:
    prep = make_stage(st, books, ix)
    if prep is None:
        break
    data, samples, stage_ids = prep
    print(f"开始训练一组：{len(stage_ids)} 本书，样本 {len(samples)} 个", flush=True)
    if train_stage(st, data, samples, T0) == "timeout":
        status = "timeout"
        break
    st["trained"] = sorted(set(st["trained"]) | set(stage_ids))
    st["stage"], st["stage_epoch"], st["stage_batch"] = [], 0, 0
    save_state(st)
    print(f"本组完成，累计已训练 {len(st['trained'])} 本", flush=True)

export(st)
save_state(st)
if status == "timeout":
    print("已达到本次 300 分钟上限，断点已保存，下次运行会接着训练", flush=True)
else:
    print("所有书籍已训练完成", flush=True)
print(f"已导出模型，本次用时 {(time.time() - T0) / 60:.1f} 分钟", flush=True)
