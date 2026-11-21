import json, os, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

RUN_SECONDS = 300 * 60   # 每次运行最多训练 300 分钟
MIN_COUNT = 1            # 出现少于 1 次的字符不纳入字表
CTX = 64                 # 一次看前 64 个字
D = 64                   # 字向量维度
HEADS = 4                # 注意力头数
LAYERS = 2               # 注意力层数
FF = 256                 # 前馈层大小
BATCH = 512
LR = 3e-4
SAVE_EVERY = 300         # 每 300 批保存一次断点
STAGE_EPOCHS = 5         # 每组新书训练的轮数
REPLAY_RATIO = 0.3       # 训练新书时混入旧书内容的比例
CHARSET_VERSION = 7      # 模型结构版本：版本不一致时从零训练
BOOK_DIR = "books"
CKPT_DIR = "ckpt"
CKPT = os.path.join(CKPT_DIR, "state.pt")
MODEL_OUT = "android/app/src/main/assets/model.json"

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(os.cpu_count() or 2)

class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Parameter(torch.randn(CTX, D) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(LAYERS):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(D),
                "attn": nn.MultiheadAttention(D, HEADS, batch_first=True),
                "ln2": nn.LayerNorm(D),
                "ff": nn.Sequential(nn.Linear(D, FF), nn.GELU(), nn.Linear(FF, D)),
            }))
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab)
        mask = torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        T = x.size(1)
        h = self.emb(x) + self.pos[:T]
        m = self.mask[:T, :T]
        for layer in self.layers:
            a = layer["ln1"](h)
            a, _ = layer["attn"](a, a, a, attn_mask=m, need_weights=False)
            h = h + a
            h = h + layer["ff"](layer["ln2"](h))
        return self.head(self.ln_f(h[:, -1, :]))

def load_books():
    books = {}
    for path in glob.glob(os.path.join(BOOK_DIR, "*.txt")):
        bid = int(os.path.splitext(os.path.basename(path))[0])
        with open(path, encoding="utf-8") as f:
            books[bid] = f.read().replace("\r", "")
    return books

def build_vocab(books, old_chars):
    import collections
    counts = collections.Counter()
    for t in books.values():
        counts.update(t)
    old_set = set(old_chars)
    new = sorted(c for c, n in counts.items() if n >= MIN_COUNT and c not in old_set)
    return old_chars + "".join(new), new

def new_state(chars):
    model = TinyLM(len(chars))
    return {
        "model": model,
        "opt": torch.optim.AdamW(model.parameters(), lr=LR),
        "chars": chars, "trained": [], "stage": [],
        "stage_epoch": 0, "stage_batch": 0, "runs": 0,
        "ver": CHARSET_VERSION,
    }

def load_or_new(books):
    chars_all, _ = build_vocab(books, "")
    if not os.path.exists(CKPT):
        return new_state(chars_all)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ck.get("ver") != CHARSET_VERSION:
        print(f"模型结构版本 {ck.get('ver')} -> {CHARSET_VERSION}，从零开始训练", flush=True)
        return new_state(chars_all)
    st = new_state(ck["chars"])
    st["model"].load_state_dict(ck["model"])
    st["opt"].load_state_dict(ck["opt"])
    st.update({k: ck[k] for k in ["trained", "stage", "stage_epoch", "stage_batch", "runs"]})
    return st

def save_state(st):
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = os.path.join(CKPT_DIR, "tmp.pt")
    torch.save({
        "model": st["model"].state_dict(),
        "opt": st["opt"].state_dict(),
        "chars": st["chars"], "trained": st["trained"], "stage": st["stage"],
        "stage_epoch": st["stage_epoch"], "stage_batch": st["stage_batch"],
        "runs": st["runs"], "ver": st["ver"],
    }, tmp)
    os.replace(tmp, CKPT)

def grow_vocab(st, new_n):
    # 字表增加时扩展输入嵌入和输出层，已有字符的参数保持不变
    if new_n == 0:
        return
    old = st["model"]
    V_old = old.head.out_features
    V_new = V_old + new_n
    model = TinyLM(V_new)
    sd = old.state_dict()
    sd_new = model.state_dict()
    for k in sd:
        if sd[k].shape == sd_new[k].shape:
            sd_new[k] = sd[k]
        elif k in ("emb.weight", "head.weight", "head.bias"):
            sd_new[k][:V_old] = sd[k]
    model.load_state_dict(sd_new)
    st["model"] = model
    st["opt"] = torch.optim.AdamW(model.parameters(), lr=LR)

def encode(text, ix):
    return np.fromiter((ix[c] for c in text if c in ix), dtype=np.int64)

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
    enc_old = np.concatenate([encode(books[b], ix) for b in old_ids]) if old_ids else np.zeros(0, np.int64)
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
    data_t = torch.from_numpy(data)
    offs = torch.arange(CTX)
    model, opt = st["model"], st["opt"]
    model.train()
    nb = (len(samples) + BATCH - 1) // BATCH
    while st["stage_epoch"] < STAGE_EPOCHS:
        rng = np.random.default_rng(1000 + st["stage_epoch"])
        perm = torch.from_numpy(samples[rng.permutation(len(samples))])
        total, seen = 0.0, 0
        t_ep = time.time()
        for bi in range(st["stage_batch"], nb):
            if time.time() - T0 > RUN_SECONDS:
                st["stage_batch"] = bi
                save_state(st)
                return "timeout"
            idx = perm[bi * BATCH:(bi + 1) * BATCH]
            xb = data_t[idx[:, None] + offs]
            yb = data_t[idx + CTX]
            logits = model(xb)
            loss = Fn.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
            seen += len(idx)
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
    model = st["model"].eval()
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    def flat(a):
        return [round(float(v), 5) for v in a.ravel()]
    out = {
        "chars": st["chars"], "ctx": CTX, "dim": D, "heads": HEADS,
        "layers": LAYERS, "ff": FF,
        "emb": flat(sd["emb.weight"]), "pos": flat(sd["pos"]),
        "head_w": flat(sd["head.weight"]), "head_b": flat(sd["head.bias"]),
        "ln_f_w": flat(sd["ln_f.weight"]), "ln_f_b": flat(sd["ln_f.bias"]),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        out[f"l{i}"] = {
            "ln1_w": flat(sd[p + "ln1.weight"]), "ln1_b": flat(sd[p + "ln1.bias"]),
            "in_w": flat(sd[p + "attn.in_proj_weight"]), "in_b": flat(sd[p + "attn.in_proj_bias"]),
            "out_w": flat(sd[p + "attn.out_proj.weight"]), "out_b": flat(sd[p + "attn.out_proj.bias"]),
            "ln2_w": flat(sd[p + "ln2.weight"]), "ln2_b": flat(sd[p + "ln2.bias"]),
            "ff1_w": flat(sd[p + "ff.0.weight"]), "ff1_b": flat(sd[p + "ff.0.bias"]),
            "ff2_w": flat(sd[p + "ff.2.weight"]), "ff2_b": flat(sd[p + "ff.2.bias"]),
        }
    with open(MODEL_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

os.makedirs(CKPT_DIR, exist_ok=True)
books = load_books()
if not books:
    raise SystemExit("books 目录里没有书，请先运行 get_corpus.py")

st = load_or_new(books)
chars, new_chars = build_vocab(books, st["chars"])
grow_vocab(st, len(new_chars))
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
