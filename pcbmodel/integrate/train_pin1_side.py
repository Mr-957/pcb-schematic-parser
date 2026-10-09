# -*- coding: utf-8 -*-
"""**(b) 学习 pin_1 侧向**：从符号裁剪图预测 pin_1 在轴的哪一端，替代 `PIN1_SIDE` 常数表。

为什么要它（2026-09-22）：`pin_geom.PIN1_SIDE` 把 `d`/`led` 标成 `ambig` 后落回 `"low"`，
而实测 `d` 只有竖直 25.6% / 水平 31.6% 符合 `"low"` ⇒ 等于每次都押少数派（val19 `d` 坐标命中
仅 **0.111**）。拟合一张 `(type,朝向)` 常数表（"a 方案"）能拿回七成，但**朝向本身也是要判的量**
（`amp_*`/`crystal_4pin` 长宽比恒 1.00 ⇒ 判不出），所以正解是**从图上学**：本模块用 GT 框裁剪
+ 朝向归一 + **纯 numpy 留一核岭**（口径同 `data/mine_orientation.py::_loo_auc`，无 sklearn）
预测"pin_1 是否在低端"，按类报 LOO AUC 与"直接押低端"的基线准确率。

⚠ 本模块只**量证据 + 存权重**，不改 `pin_geom`（接线另一步，需 paired A/B）。

跑（本项目 env 任意，只要有 numpy+PIL）：
  python -m pcbmodel.integrate.train_pin1_side                  # 200 例，按类报 AUC/acc
  python -m pcbmodel.integrate.train_pin1_side --out pcbmodel/runs/stage0/pin1_side.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

IMG_SIDE = 12          # 特征分辨率（12x12=144 维，够核岭；200 例量级不必更大）
PIN1_LOW = 0           # 标签：0 = pin_1 在低端（下/左），1 = 在高端（上/右）


def _axis(bbox, ctype: str) -> str:
    """轴线判据与 `pin_geom.pin_axis` 一致（`crystal_2pin` 长宽比反用）。"""
    x1, y1, x2, y2 = bbox
    vert = (y2 - y1) >= (x2 - x1)
    if ctype == "crystal_2pin":
        vert = not vert
    return "vert" if vert else "horiz"


def _feat(img: np.ndarray, bbox, axis: str) -> np.ndarray:
    """符号裁剪 → 朝向归一（轴恒竖直）→ 12×12 灰度 → 去均值单位方差。纯 numpy。"""
    from PIL import Image
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, max(x1 + 1, x2))
    y1, y2 = max(0, y1), min(h, max(y1 + 1, y2))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(IMG_SIDE * IMG_SIDE, dtype=np.float64)
    im = Image.fromarray(crop).convert("L")
    if axis == "horiz":                       # 转成"轴竖直"，消掉横竖差异
        im = im.transpose(Image.ROTATE_90)
    im = im.resize((IMG_SIDE, IMG_SIDE), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float64).reshape(-1) / 255.0
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-6 else a


def double_tables(limit: int = 0) -> dict:
    """**同一份数据、同一个脚本**产出两张表（防"帧/口径"误读，2026-09-22 订正）：

    - `abs[type]` = 绝对方向 U/D/L/R（**target y-up**：U = y 大；引用务必标帧！）
    - `low[type]` = **轴相对低端** [low, high]（先按长宽比判轴，再看 pin_1 是否在轴向坐标更小的一端）

    返回 `{"n": {type: n}, "abs": {type: Counter}, "low": {type: [low, high]}}`。
    """
    ds = ROOT / "Dataset300"
    tg = ROOT / "赛题六公开数据集" / "200_train_cases"
    tids = sorted(d.name for d in tg.iterdir() if d.is_dir())
    out = {"n": defaultdict(int), "abs": defaultdict(Counter),
           "low": defaultdict(lambda: [0, 0])}
    for cid in tids:
        tfile = next(iter(sorted(tg.joinpath(cid).glob("*_target_named.json"))), None)
        if tfile is None:
            continue
        t = json.loads(tfile.read_text(encoding="utf-8"))
        for k, c in (t.get("components") or {}).items():
            pd = (t.get("pins") or {}).get(k) or {}
            if len(pd) != 2 or "pin_1" not in pd:
                continue
            p1 = (pd["pin_1"].get("point") or {})
            if "x" not in p1:
                continue
            ty = c.get("type", "?")
            x1, y1, x2, y2 = [float(v) for v in c["bbox"]]
            px, py = float(p1["x"]), float(p1["y"])
            dx, dy = px - (x1 + x2) / 2, py - (y1 + y2) / 2
            d_abs = ("U" if abs(dy) >= abs(dx) and dy > 0
                     else "D" if abs(dy) >= abs(dx) else "L" if dx < 0 else "R")
            ax = _axis([x1, y1, x2, y2], ty)
            t1 = ((py - y1) / max(1e-6, y2 - y1) if ax == "vert"
                  else (px - x1) / max(1e-6, x2 - x1))
            out["n"][ty] += 1
            out["abs"][ty][d_abs] += 1
            out["low"][ty][0 if t1 < 0.5 else 1] += 1
        if limit and sum(out["n"].values()) >= limit:
            break
    return out


def selftest() -> int:
    """**双表单测**：同一份数据必须同时产出 (U/D/L/R) 与 LOW 占比，且不变量成立。

    锁三条（2026-09-22 实测）：① 常数表类（r/c/l/fuse/crystal_2pin）低端 **100%**；
    ② **d 低端 ≈29%**（= 账本"竖直 25.6%/水平 31.6%"，**不是** 71%）；
    ③ c 的绝对方向 **D 最多 ≈78%**（y-up 帧）—— 与图像 y-down 表的"上 78%"互为翻转。
    """
    tb = double_tables()
    for ty in ("r", "c", "l", "fuse", "crystal_2pin"):
        lo, hi = tb["low"][ty]
        assert hi == 0 and lo > 0, f"{ty} 低端应 100%：{lo}/{hi}"
    lo, hi = tb["low"]["d"]
    share = lo / (lo + hi)
    assert 0.25 <= share <= 0.33, f"d 低端占比应 ≈29%，实测 {share:.1%}"
    a = tb["abs"]["c"]
    assert max(a, key=a.get) == "D" and a["D"] / sum(a.values()) > 0.70, dict(a)
    print(f"[pin1_side] 双表单测 OK：常数表类 100% 低端｜d 低端 {share:.1%}"
          f"｜c 绝对方向 D 占 {a['D'] / sum(a.values()):.0%}（y-up 帧）")
    return 0


def collect(limit: int = 0):
    """→ `X (n,144)`, `y (n,)`（**1 = pin_1 在高端**，0 = 低端）, `types (n,)`。数据 = 200 例官方 target。"""
    ds = ROOT / "Dataset300"
    tg = ROOT / "赛题六公开数据集" / "200_train_cases"
    tids = sorted(d.name for d in tg.iterdir() if d.is_dir())
    # 图：Dataset300/<id>/<id>-*.png（y-down）；target：200_train_cases（y-up）
    X, y, types, cases = [], [], [], []
    for cid in tids:
        tfile = next(iter(sorted(tg.joinpath(cid).glob("*_target_named.json"))), None)
        png = next(iter(sorted(ds.joinpath(cid).glob("*.png"))), None) if (ds / cid).is_dir() else None
        if tfile is None or png is None:
            continue
        t = json.loads(tfile.read_text(encoding="utf-8"))
        from PIL import Image
        with Image.open(png) as im:
            img = np.asarray(im.convert("L"))
        H = img.shape[0]
        for k, c in (t.get("components") or {}).items():
            pd = (t.get("pins") or {}).get(k) or {}
            if len(pd) != 2 or "pin_1" not in pd or "pin_2" not in pd:
                continue
            p1 = (pd["pin_1"].get("point") or {})
            if "x" not in p1:
                continue
            ty = c.get("type", "?")
            bx, by = [float(v) for v in c["bbox"]], None       # y-up
            ax = _axis(bx, ty)
            x1, y1, x2, y2 = bx
            if ax == "vert":
                t1 = (float(p1["y"]) - y1) / max(1e-6, y2 - y1)
            else:
                t1 = (float(p1["x"]) - x1) / max(1e-6, x2 - x1)
            # y-up → y-down 像素框（§八 风险 12：跨帧必须显式转换）
            dbox = [x1, H - y2, x2, H - y1]
            X.append(_feat(img, dbox, ax))
            y.append(0 if t1 < 0.5 else 1)                     # 0 = 低端
            types.append(ty)
            cases.append(cid)
        if limit and len(X) >= limit:
            break
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64),
            np.asarray(types), np.asarray(cases))


def loo_scores(K: np.ndarray, yv: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """留一核岭（hat 矩阵闭式解）—— 口径同 `data/mine_orientation.py::_loo_auc`。"""
    n = len(yv)
    P = np.linalg.inv(K + lam * n * np.eye(n))
    A = P @ yv.astype(np.float64)
    return yv.astype(np.float64) - A / np.clip(np.diag(P), 1e-9, None)


def auc(yv: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[yv == 1], s[yv == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def loo_by_case(X: np.ndarray, yv: np.ndarray, cases: np.ndarray,
                lam: float = 1e-2) -> np.ndarray:
    """**按图留一**（held-out case 整体不进训练）→ held-out 预测；不可训的折留 nan。

    ⚠ 样本级 LOO（`loo_scores`）**有泄漏**：同一张图的多个元件共享符号风格/朝向分布，
    同时进训练与测试 ⇒ AUC 虚高或虚低都可能（2026-09-22 实测 led：样本级 0.677 vs 按图 0.962）。
    """
    pred = np.full(len(yv), np.nan)
    for c in sorted(set(cases.tolist())):
        m = cases == c
        Xt, yt = X[~m], yv[~m].astype(float)
        if len(yt) < 10 or len(set(yt.tolist())) < 2:
            continue
        n = len(yt)
        K = Xt @ Xt.T / Xt.shape[1]
        A = np.linalg.solve(K + lam * n * np.eye(n), yt)
        pred[m] = (X[m] @ Xt.T / Xt.shape[1]) @ A
    return pred


def per_type_report(X: np.ndarray, yv: np.ndarray, types: np.ndarray,
                    cases: np.ndarray, lam: float = 1e-2) -> list[tuple]:
    """每类**单独建模 + 按图留一** → `(type, n, n_case, auc, acc, low_share)`。"""
    rows = []
    for ty in sorted(set(types.tolist()), key=lambda z: -int((types == z).sum())):
        m = types == ty
        n = int(m.sum())
        if n < 30:
            continue
        yy, cc = yv[m], cases[m]
        low = float((yy == 0).mean())
        if low in (0.0, 1.0):
            # 单类（r/c/l/crystal…）⇒ AUC 定义不了，且它们样本最多：**先跳过**，
            # 否则按图留一会对每张图解一个 ~1500×1500 的核岭，白烧几分钟。
            rows.append((ty, n, len(set(cc.tolist())), float("nan"), float("nan"), low))
            continue
        p = loo_by_case(X[m], yy, cc, lam=lam)
        ok = ~np.isnan(p)
        if ok.sum() < 10 or len(set(yy[ok].tolist())) < 2:
            rows.append((ty, n, len(set(cc.tolist())), float("nan"), float("nan"), low))
            continue
        thr = float(yy[ok].mean())            # 阈值 = 先验（不平衡下不校准会系统性偏低）
        rows.append((ty, n, len(set(cc.tolist())), auc(yy[ok], p[ok]),
                     float(((p[ok] > thr).astype(int) == yy[ok]).mean()), low))
    return rows


def box_corner_report(limit: int = 0) -> list[tuple]:
    """`box` 的 **pin_1「角脚」语义**：不看"轴向端"，改看 pin_1 相对 bbox 中心的 **x/y 符号**
    （两个独立二分类）⇒ 回答"0.872 站不站得住"（IC 的 pin_1 本来就是角脚，不是轴端）。
    返回 `[(label_name, n, n_case, auc, acc)]`。
    """
    ds = ROOT / "Dataset300"
    tg = ROOT / "赛题六公开数据集" / "200_train_cases"
    X, Lx, Ly, cases = [], [], [], []
    for cid in sorted(d.name for d in tg.iterdir() if d.is_dir()):
        tfile = next(iter(sorted(tg.joinpath(cid).glob("*_target_named.json"))), None)
        png = next(iter(sorted(ds.joinpath(cid).glob("*.png"))), None) if (ds / cid).is_dir() else None
        if tfile is None or png is None:
            continue
        t = json.loads(tfile.read_text(encoding="utf-8"))
        from PIL import Image
        with Image.open(png) as im:
            img = np.asarray(im.convert("L"))
        H = img.shape[0]
        for k, c in (t.get("components") or {}).items():
            if c.get("type") != "box":
                continue
            pd = (t.get("pins") or {}).get(k) or {}
            if len(pd) != 2 or "pin_1" not in pd:
                continue
            p1 = pd["pin_1"].get("point") or {}
            if "x" not in p1:
                continue
            x1, y1, x2, y2 = [float(v) for v in c["bbox"]]
            X.append(_feat(img, [x1, H - y2, x2, H - y1], _axis([x1, y1, x2, y2], "box")))
            Lx.append(1 if float(p1["x"]) > (x1 + x2) / 2 else 0)
            Ly.append(1 if float(p1["y"]) > (y1 + y2) / 2 else 0)
            cases.append(cid)
        if limit and len(X) >= limit:
            break
    if len(X) < 30:
        return []
    Xa, ca = np.asarray(X), np.asarray(cases)
    out = []
    for name, yv in (("box pin_1 x>中线", np.asarray(Lx)), ("box pin_1 y>中线", np.asarray(Ly))):
        p = loo_by_case(Xa, yv, ca)
        ok = ~np.isnan(p)
        if ok.sum() < 10 or len(set(yv[ok].tolist())) < 2:
            out.append((name, len(yv), len(set(ca.tolist())), float("nan"), float("nan")))
            continue
        thr = float(yv[ok].mean())
        out.append((name, len(yv), len(set(ca.tolist())), auc(yv[ok], p[ok]),
                    float(((p[ok] > thr).astype(int) == yv[ok]).mean())))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_pin1_side", description="学习 pin_1 侧向")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--out", default=None, help="存权重 npz（给 pin_geom 接线用）")
    ap.add_argument("--selftest", action="store_true", help="双表单测（不训练）")
    ap.add_argument("--cv", default="by-case", choices=["by-case", "sample"],
                    help="交叉验证口径：by-case = **按图留一**（默认，无泄漏）；"
                         "sample = 样本级留一（⚠ 同图多例泄漏，只作历史对照）")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    X, y, types, cases = collect(a.limit)
    tb = double_tables(a.limit)
    print(f"[pin1_side] 样本 {len(X)}（{len(set(cases.tolist()))} 例）")
    print("\n=== 双表（同一份数据；⚠ abs 方向是 target **y-up** 帧，引用必须标帧）===")
    print(f"{'type':14s} {'n':>5s} | {'绝对方向 U/D/L/R':28s} | {'低端占比':>7s}")
    for ty in sorted(tb["n"], key=lambda z: -tb["n"][z]):
        n = tb["n"][ty]
        if n < 10:
            continue
        aa, (lo, hi) = tb["abs"][ty], tb["low"][ty]
        s = " ".join(f"{k}:{aa[k] / n:.0%}" for k in ("U", "D", "L", "R") if aa[k])
        print(f"{ty:14s} {n:5d} | {s:28s} | {lo / n:7.1%}")
    K = X @ X.T / X.shape[1]                    # 线性核（0.5.5.2 实测线性 0.983，够用）
    # ⚠ 全类这一行**固定用样本级留一**（`loo_scores`）：3600 样本做按图留一要 198 次
    #   3600×3600 求解，纯属烧机时；分类表（下面）才用 `--cv` 选的口径。**别引全类这一行。**
    pred = loo_scores(K, y, lam=a.lam)
    print(f"\n{'type':16s} {'n':>4s} {'押多数端acc':>11s} {'模型LOO acc':>11s} {'LOO AUC':>8s}"
          f"  ⚠阈值=先验")
    rows = []
    for t in sorted(set(types.tolist()), key=lambda z: -int((types == z).sum())):
        m = types == t
        n = int(m.sum())
        if n < 20:
            continue
        base = max(float((y[m] == 0).mean()), float((y[m] == 1).mean()))
        thr = float((y[m] == 1).mean())          # 阈值取先验；否则不平衡标签下 acc 系统性偏低
        acc = float(((pred[m] > thr).astype(int) == y[m]).mean())
        rows.append((t, n, base, acc, auc(y[m], pred[m])))
        print(f"{t:16s} {n:4d} {base:9.1%} {acc:11.1%} {rows[-1][4]:8.3f}")
    allacc = float(((pred > float((y == 1).mean())).astype(int) == y).mean())
    print(f"\n全类（⚠ 样本级留一，泄漏口径，仅参考）：押多数端 "
          f"{max((y==0).mean(),(y==1).mean()):.1%} ｜ acc {allacc:.1%} ｜ AUC {auc(y, pred):.3f}")
    print(f"\n=== 每类单独建模 + 按图留一（cv={a.cv}）—— 与上表口径不同，别混引 ===")
    print(f"{'type':14s} {'n':>5s} {'图':>4s} {'AUC':>7s} {'acc':>7s} {'低端占比':>9s}")
    for ty, n, nc, au, ac, low in per_type_report(X, y, types, cases, lam=a.lam):
        print(f"{ty:14s} {n:5d} {nc:4d} {au:7.3f} {ac:7.1%} {low:9.1%}")
    br = box_corner_report()
    if br:
        print(f"\n=== box「角脚」语义（pin_1 相对 bbox 中心的符号，两个二分类）===")
        print(f"{'label':18s} {'n':>5s} {'图':>4s} {'AUC':>7s} {'acc':>7s}")
        for name, n, nc, au, ac in br:
            print(f"{name:18s} {n:5d} {nc:4d} {au:7.3f} {ac:7.1%}")
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, X=X, y=y, types=types, side=IMG_SIDE, lam=a.lam)
        print(f"[pin1_side] 权重/特征表 -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
