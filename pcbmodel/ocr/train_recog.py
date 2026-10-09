# -*- coding: utf-8 -*-
"""P1 阶段 1.2：OCR **识别**模型训练（CRNN: CNN+BiLSTM + CTC）。

数据：`runs/ocr/recog_v1/{train,val}/`（labels.txt：相对路径 TAB 标签；samples.json：类别映射）。
输入：灰度裁剪图，高度归一 32，宽度按比例保留（可变）；batch 内右补齐。
输出：`runs/ocr/recog_v1_train/`（best.pt + vocab.json + last.pt + metrics.json）。

用法（pcbmodel env）：
  python -m pcbmodel.ocr.train_recog --out pcbmodel/runs/ocr/recog_v1_train \
      --device 0 --epochs 60 --batch 128
  python -m pcbmodel.ocr.train_recog --predict ...   # 推理/按类评估

选择口径（--metric cls_macro 默认）：按类精确宏平均 —— char_acc 未加权会忽略 name
（val 仅 3%），而下游要的是整串精确（位号错一字符即丢 compF1）。训练侧裁剪抖动
（--jitter 默认开）对齐推理期 1.1e 预测框误差（~15% 身高）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

H = 32          # 归一高度
NUM_CHAR = 512  # 词表上限


# ---------------- 数据 ----------------

def load_split(split_dir: Path):
    """返回 [(img: HWC uint8, label: str, cls: str), ...]"""
    labels = (split_dir / "labels.txt").read_text(encoding="utf-8").splitlines()
    cls_map = {}
    sj = split_dir / "samples.json"
    if sj.exists():
        cls_map = {s["file"]: s["class"] for s in json.loads(sj.read_text(encoding="utf-8"))}
    items = []
    for ln in labels:
        if not ln.strip():
            continue
        rel, label = ln.split("\t", 1)
        items.append((rel, label, cls_map.get(rel, "")))
    return items


def build_vocab(items, min_count: int = 1):
    cnt = Counter()
    for _, label, _ in items:
        cnt.update(label)
    chars = sorted(c for c, n in cnt.items() if n >= min_count)
    if len(chars) > NUM_CHAR:
        chars = chars[:NUM_CHAR]           # 高频前 N
    idx = {c: i for i, c in enumerate(chars)}
    return chars, idx                       # blank = len(chars)


def _is_vertical(im: Image.Image, thr: int = 32) -> bool:
    """竖排文字判定（Claude §五 D11，修订版）。

    判据：h/w≥1.4 且**成块墨迹行带 ≥2 且带间有真实间隙**。
    教训：低分辨率单字（如正立 '2' 在 23×15 框）的笔画会被阈值拆成多条 1px 高的
    散碎行带（4 带、各 1px）——那**不是竖排**，是正立单字（占高箱 78%），绝不能转。
    真竖排（'123'/'CLK'/'B5'）的行带是 ≥2px 高的成块、带间间隙 ≥2 行。
    """
    a = np.asarray(im.convert("L"), dtype=np.uint8)
    dark = (a < thr).any(axis=1)
    # 原始行带
    bands = []
    in_band = False
    for i, d in enumerate(dark):
        if d:
            if not in_band:
                bands.append([i, i])
                in_band = True
            else:
                bands[-1][1] = i
        else:
            in_band = False
    # 合并间隙 ≤1 行的近距带（同一字形的笔画碎片）
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= 1:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    big = [b for b in merged if b[1] - b[0] + 1 >= 2]   # 只认成块带
    if len(big) < 2:
        return False
    gaps = [big[i + 1][0] - big[i][1] - 1 for i in range(len(big) - 1)]
    return max(gaps) >= 2                                # 带间真实间隙


def load_img(path: Path, rotate: bool = True) -> np.ndarray:
    im = Image.open(path).convert("L")
    w, h = im.size
    # 竖排文字修法（Claude §五 D11）：h/w≥1.4 且成块墨迹行带≥2 → 顺时针旋转 270°。
    # 机理：高度归一到 32 + 两次 (2,2) 池化 ⇒ CTC 时间步 T≈8/L；竖排多字 L≥3 时 T<L
    # 不可行，zero_infinity 静默置 0（学不到却计分母）。单字高箱不转。
    # ⚠ 第 4 轮（2026-09-12，Doubao）：旋转被实测为负贡献（端到端 11.61 < baseline 14.76），
    # 训练/推理统一 `rotate=False`（回滚）；竖排样本由 `drop_vertical` 在数据集层丢弃，
    # 不给 CTC 喂结构上不可读的垃圾标签。
    if rotate and h >= 1.4 * w and _is_vertical(im):
        im = im.transpose(Image.ROTATE_270)
        w, h = im.size
    nh = H
    nw = max(1, round(w * nh / h))
    im = im.resize((nw, nh), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a                                 # (H, nw)


def jitter_crop(img: np.ndarray, max_d: int = 4, scale: tuple = (0.95, 1.05)) -> np.ndarray:
    """训练侧裁剪抖动（对推理期预测框误差的鲁棒性，Claude 建议 ②）。

    推理吃 1.1e 预测框裁剪（误差 ~15% 身高 ≈ ±3px）；训练吃 GT 框裁剪（像素级对齐）。
    加 ±4px 平移 + 宽向 ±5% 缩放，把分布对齐。
    """
    h, w = img.shape
    im = Image.fromarray((img * 255.0).astype(np.uint8))
    s = random.uniform(*scale)
    nw = max(1, round(w * s))
    if nw != w:
        im = im.resize((nw, h), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    pad = max_d
    ap = np.pad(a, ((pad, pad), (pad, pad)), mode="constant")
    dy = random.randint(-pad, pad)
    dx = random.randint(-pad, pad)
    return ap[pad + dy: pad + dy + h, pad + dx: pad + dx + nw]


class RecogDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: Path, idx: dict, preload: bool = True,
                 class_balanced: bool = False, jitter: bool = False,
                 max_len: int = 64, rotate: bool = True,
                 drop_vertical: bool = False,
                 lowercase_boost: float = 0.0):
        """lowercase_boost：>1 时，label 含小写字母的样本采样权重乘该倍数。

        背景（Claude probe_recog_gap，2026-09-17）：val 里含小写的位号只有 41 条
        （designator 的 1.4%）却贡献 ~34% 的错误（0.5610 vs 全大写 0.9883）——
        模型输出塌缩到大写（98.6% 样本是大写）。词表含小写 a-z，能输出，只是采样
        占比太低。字符级过采样 = 权重乘数，不动数据文件。
        """
        self.items = load_split(split_dir)
        self.idx = idx
        self.blank = len(idx)
        self.split_dir = split_dir
        self.max_len = max_len
        self.jitter = jitter
        self.rotate = rotate
        self.images = None
        self.ws = []
        if drop_vertical:
            # 第 4 轮（2026-09-12，Doubao）：竖排（_is_vertical）CTC 结构上不可读，
            # 旋转救不了（D11）且旋转端到端实测负贡献——直接丢弃，不给模型喂垃圾标签。
            # 当前修订版判定：train 215 / val 27（Claude 引用的 430/49 是修订前旧口径；
            # 修订版修掉了低分辨率正立单字误判）。L<3 竖排一并丢弃（不旋转时同样不可读）。
            kept = []
            for rel, label, cls in self.items:
                im = Image.open(split_dir / rel).convert("L")
                if im.size[1] >= 1.4 * im.size[0] and _is_vertical(im):
                    continue
                kept.append((rel, label, cls))
            self.items = kept
        if preload:
            self.images = [load_img(split_dir / rel, rotate=self.rotate)
                           for rel, _, _ in self.items]
            self.ws = [im.shape[1] for im in self.images]
        # 采样权重 = 类别平衡 × 小写 boost（两者都可独立开关）
        w = [1.0] * len(self.items)
        if class_balanced:
            freq = Counter(cls for _, _, cls in self.items if cls)
            if freq:
                cw = {cls: 1.0 / math.sqrt(n) for cls, n in freq.items()}
                fb = min(cw.values())          # 未知/空类别兜底 = 多数类的低权重
            else:
                cw, fb = {}, 1.0
            w = [cw.get(cls, fb) for _, _, cls in self.items]
        if lowercase_boost and lowercase_boost != 1.0:
            w = [wi * (lowercase_boost if any(c.islower() for c in label) else 1.0)
                 for wi, (_, label, _) in zip(w, self.items)]
        self.weights = (torch.tensor(w, dtype=torch.double)
                        if any(x != 1.0 for x in w) else None)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label, cls = self.items[i]
        if self.images is None:
            img = load_img(self.split_dir / rel, rotate=self.rotate)
        else:
            img = self.images[i]
        if self.jitter:
            img = jitter_crop(img)
        lab = [self.idx[c] for c in label if c in self.idx]
        return img, torch.tensor(lab, dtype=torch.long), cls, rel


def collate(batch):
    imgs, labs, clss, rels = zip(*batch)
    W = max(im.shape[1] for im in imgs)
    B = len(imgs)
    x = torch.zeros(B, 1, H, W)
    for i, im in enumerate(imgs):
        x[i, 0, :, : im.shape[1]] = torch.from_numpy(im)
    target = torch.cat([l for l in labs])
    tl = torch.tensor([len(l) for l in labs], dtype=torch.long)
    return x, target, tl, list(clss), list(rels)


# ---------------- 模型 ----------------

class CRNN(nn.Module):
    """CNN(高度压缩到 1) + BiLSTM×2 + 线性。输入 (B,1,32,W) → logits (W/4, B, C)。"""

    def __init__(self, n_class: int, hidden: int = 256):
        super().__init__()
        def cbn(i, o, k=3, s=1, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(
            cbn(1, 64), nn.MaxPool2d((2, 2)),                       # H16
            cbn(64, 128), nn.MaxPool2d((2, 2)),                     # H8
            cbn(128, 256), cbn(256, 256), nn.MaxPool2d((2, 1)),     # H4
            cbn(256, 512), cbn(512, 512), nn.MaxPool2d((2, 1)),     # H2
            cbn(512, 512), nn.AdaptiveAvgPool2d((1, None)),         # H1
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden, bidirectional=True, num_layers=2,
                    dropout=0.2, batch_first=True),
        )
        self.fc = nn.Linear(hidden * 2, n_class)

    def forward(self, x):
        f = self.cnn(x)                        # (B,512,1,W/4)
        f = f.squeeze(2).transpose(1, 2)       # (B, W/4, 512)
        r, _ = self.rnn(f)
        return self.fc(r).transpose(0, 1)      # (T, B, C)


def greedy_decode(logits: torch.Tensor) -> list[str]:
    """logits (T,B,C) → 每样本 CTC 贪心解码串（不含 blank）。"""
    T, B, C = logits.shape
    ids = logits.argmax(-1).cpu().numpy()      # (T,B)
    blank = C - 1
    out = []
    for b in range(B):
        s = []
        prev = blank
        for t in range(T):
            c = int(ids[t, b])
            if c != blank and c != prev:
                s.append(c)
            prev = c
        out.append(s)
    return out


# ---------------- 训练 ----------------

def run_train(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    data = Path(args.data)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tr_items = load_split(data / "train")
    chars, idx = build_vocab(tr_items, min_count=args.min_char_count)
    blank = len(chars)
    print(f"[train_recog] vocab {len(chars)} 字符（blank={blank}）; train {len(tr_items)} 样本")
    (Path(args.out) / "vocab.json").write_text(
        json.dumps({"chars": chars, "blank": blank}), encoding="utf-8")

    tr = RecogDataset(data / "train", idx, class_balanced=args.class_balanced,
                      jitter=args.jitter, rotate=args.rotate,
                      drop_vertical=args.drop_vertical,
                      lowercase_boost=args.lowercase_boost)
    va = RecogDataset(data / "val", idx, rotate=args.rotate)
    print(f"[train_recog] 训练样本 {len(tr_items)}（drop_vertical 后 {len(tr)}）")
    sampler = torch.utils.data.WeightedRandomSampler(
        tr.weights, len(tr), replacement=True) if tr.weights is not None else None
    tr_dl = torch.utils.data.DataLoader(
        tr, batch_size=args.batch, shuffle=sampler is None,
        sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=True)
    va_dl = torch.utils.data.DataLoader(
        va, batch_size=args.batch, shuffle=False, collate_fn=collate,
        num_workers=2, pin_memory=True)

    model = CRNN(blank + 1, hidden=args.hidden).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_recog] CRNN 参数 {n_params/1e6:.2f}M")

    crit = nn.CTCLoss(blank=blank, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1)

    best = {"acc": 0.0, "cer": 1.0, "epoch": -1}
    Path(args.out).mkdir(parents=True, exist_ok=True)

    def evaluate() -> dict:
        model.eval()
        cls_c = defaultdict(lambda: [0, 0, 0])   # cls -> [right, wrong, total]
        tot_chars = tot_right = 0
        low_right = low_tot = 0                  # label 含小写的子集（probe_recog_gap 病根）
        with torch.no_grad():
            for x, target, tl, clss, rels in va_dl:
                logits = model(x.to(dev))
                preds = greedy_decode(logits)
                # 逐样本对齐 target
                off = 0
                for p, l, cls in zip(preds, tl.tolist(), clss):
                    lab = [int(t) for t in target[off: off + l]]
                    off += l
                    ref = "".join(chars[c] for c in lab)
                    hyp = "".join(chars[c] for c in p if c < blank)
                    tot_chars += len(ref)
                    tot_right += sum(a == b for a, b in zip(ref, hyp))
                    if any(c.islower() for c in ref):
                        low_tot += 1
                        low_right += 1 if hyp == ref else 0
                    if hyp == ref:
                        cls_c[cls][0] += 1
                    else:
                        cls_c[cls][1] += 1
                    cls_c[cls][2] += 1
        model.train()
        char_acc = tot_right / max(tot_chars, 1)
        cls_acc = {k: v[0] / max(v[2], 1) for k, v in cls_c.items()}
        macro = (sum(cls_acc[k] for k in cls_acc if cls_c[k][2] > 0)
                 / max(len(cls_acc), 1))
        return {"char_acc": char_acc, "cls_macro": macro,
                "cls_acc": cls_acc, "cls_n": {k: v[2] for k, v in cls_c.items()},
                "lowercase_acc": low_right / max(low_tot, 1),
                "lowercase_n": low_tot}

    sel = args.metric                                  # 选择口径（① Claude 建议）
    best = {"v": -1.0, "epoch": -1}
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = tot_n = 0
        for x, target, tl, *_ in tr_dl:
            x, target, tl = x.to(dev), target.to(dev), tl.to(dev)
            logits = model(x)                  # (T,B,C)
            T = logits.shape[0]
            il = torch.full((x.shape[0],), T, dtype=torch.long, device=dev)
            loss = crit(F.log_softmax(logits, 2), target, il, tl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * x.shape[0]
            tot_n += x.shape[0]
        m = evaluate()
        tag = ""
        if m[sel] > best["v"]:
            best = {"v": m[sel], "epoch": ep, "metric": sel}
            torch.save({"model": model.state_dict(), "vocab": chars,
                        "args": vars(args), "metric": m}, Path(args.out) / "best.pt")
            tag = " ★best"
        if ep % args.log_every == 0 or ep == args.epochs:
            cls = " ".join(f"{k}:{v:.3f}" for k, v in sorted(m["cls_acc"].items()))
            print(f"  ep{ep:>3} loss {tot_loss/max(tot_n,1):.4f} "
                  f"{sel} {m[sel]:.4f} (char {m['char_acc']:.4f} "
                  f"lower {m['lowercase_acc']:.4f}/{m['lowercase_n']}) | {cls}{tag}")
        if m[sel] >= args.target_acc:
            print(f"[train_recog] 提前达标 {args.target_acc}，停")
            break

    torch.save({"model": model.state_dict(), "vocab": chars,
                "args": vars(args), "metric": m}, Path(args.out) / "last.pt")
    (Path(args.out) / "metrics.json").write_text(
        json.dumps({"best": best, "final": m}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[train_recog] done. best ep{best['epoch']} {sel}={best['v']:.4f} "
          f"-> {Path(args.out).resolve()}")


def run_predict(args) -> None:
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(args.out) / "best.pt", map_location="cpu")
    chars = ck["vocab"]
    blank = len(chars)
    idx = {c: i for i, c in enumerate(chars)}
    model = CRNN(blank + 1, hidden=ck["args"].get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    ds = RecogDataset(Path(args.data) / "val", idx, rotate=args.rotate)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    cls_c = defaultdict(lambda: [0, 0])
    bad: list[tuple] = []
    with torch.no_grad():
        for x, target, tl, clss, rels in dl:
            logits = model(x.to(dev))
            preds = greedy_decode(logits)
            off = 0
            for p, l, cls, rel in zip(preds, tl.tolist(), clss, rels):
                lab = [int(t) for t in target[off: off + l]]
                off += l
                ref = "".join(chars[c] for c in lab)
                hyp = "".join(chars[c] for c in p if c < blank)
                cls_c[cls][1] += 1
                if hyp == ref:
                    cls_c[cls][0] += 1
                elif len(bad) < args.show_bad:
                    bad.append((rel, ref, hyp, cls))
    tot = sum(v[1] for v in cls_c.values())
    right = sum(v[0] for v in cls_c.values())
    print(f"[predict] val 串准确率 {right/tot:.4f} ({right}/{tot})")
    for k in sorted(cls_c):
        v = cls_c[k]
        print(f"  {k:<10} {v[0]}/{v[1]} = {v[0]/max(v[1],1):.4f}")
    for rel, ref, hyp, cls in bad:
        print(f"  ✗ {rel} [{cls}] ref={ref!r} hyp={hyp!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_recog")
    ap.add_argument("--data", default=str(ROOT / "pcbmodel" / "runs" / "ocr" / "recog_v1"))
    ap.add_argument("--out", default=str(ROOT / "pcbmodel" / "runs" / "ocr" / "recog_v1_train"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--class-balanced", dest="class_balanced",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="类别平衡采样（--no-class-balanced 关闭）")
    ap.add_argument("--jitter", dest="jitter",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="训练侧裁剪抖动 ±4px/±5%%（--no-jitter 关闭）")
    ap.add_argument("--rotate", dest="rotate",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="load_img 是否旋转竖排（第 3 轮行为；第 4 轮回滚用 --no-rotate）")
    ap.add_argument("--drop-vertical", dest="drop_vertical",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="训练集丢弃竖排样本（第 4 轮：CTC 结构上不可读，D11）")
    ap.add_argument("--lowercase-boost", type=float, default=0.0,
                    help="label 含小写字母的样本采样权重乘数（0=关；probe_recog_gap 病根）")
    ap.add_argument("--metric", choices=["char_acc", "cls_macro", "lowercase_acc"],
                    default="cls_macro",
                    help="best 选择口径：cls_macro=按类精确宏平均（推荐，不忽略稀类）"
                         "；lowercase_acc=val 含小写样本串准确率（救小写专用）")
    ap.add_argument("--min-char-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--target-acc", type=float, default=0.999)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--show-bad", type=int, default=20)
    args = ap.parse_args(argv)
    (run_predict if args.predict else run_train)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
