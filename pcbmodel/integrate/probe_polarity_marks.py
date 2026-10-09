# -*- coding: utf-8 -*-
"""**极性标记探针（跨工具用）**：不比对整体像素，只找"垂直于轴的整条横杠/侧边"这类**几何标记**。

动机（2026-09-22）：同分布内 pin_1 侧向的按图留一 AUC = d 0.933 / led 0.962，但**跨工具不可判**
（10gt 只有 d=3/led=5，AUC 分辨率 1/3、1/6）。所以换一条**工具无关**的路：把"极性"落成可定位的
标记 —— 二极管阴极横杠、LED 侧边/箭头、IC 的 pin-1 角标 —— 用**位置统计量**（每个轴向位置的
墨迹覆盖率）判定它落在哪一端，再看"**标记端 == pin_1 端**"这个**比例**在两份数据源上是否一致。
比例对样本量远不敏感（8 例也能看方向），这正是 AUC 做不到的。

**口径**（防帧错，§八 风险 12）：
  - 裁剪后在 **target y-up 帧**里规整：`arr[0]` = 轴的**低端**（竖轴=下、横轴=左）；
  - `cov[i]` = 第 i 个轴向位置上、**垂直于轴**那条线的墨迹占比（0..1）；
  - 取 `i* = argmax(cov)`，`strength = cov[i*]`；只有 `strength ≥ --min-strength` 才算"有横杠"。

跑（任意有 numpy+PIL 的 env；本地即可）：
  python -m pcbmodel.integrate.probe_polarity_marks                 # train vs 10gt 对比表
  python -m pcbmodel.integrate.probe_polarity_marks --sheet out.png # 另存目视拼图
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pcbmodel.integrate.train_pin1_side import _axis      # 轴判据与侧向表**同一把尺子**


def axis_profile(img: np.ndarray, bbox_up, ctype: str, pad: int = 0) -> np.ndarray | None:
    """符号裁剪 → （轴规整后的）**暗像素矩阵**，行 = 轴向、`arr[0]` = 轴低端。

    `pad`：向四周多留的像素（2026-09-22 新增）。**为什么需要**：官方 `bbox` 是**符号本体、
    不含引脚**，而 **LED 的箭头常画在本体之外**、IC 的 pin-1 圆点偶尔也贴边 ⇒ pad=0 时这些
    标记**在裁剪里根本不存在**（很可能就是 led 跨工具不稳的根因）。
    """
    # ⚠⚠ **必须先转灰度**（2026-09-22 真实数据崩溃的根因）：探针里读图是 `convert("L")`，
    #   而 `pin_side` 直接吃 `load_sample().image_orig`（**RGB**）⇒ 不转的话 `dark` 是 3 维，
    #   `cov[i]` 就是向量，`float(cov[i])` 抛 `TypeError: only 0-dimensional arrays ...`。
    if img.ndim == 3:
        img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
    x1, y1, x2, y2 = (float(v) for v in bbox_up)          # y-up
    H, W = img.shape[:2]
    t, b = int(round(H - y2)) - pad, int(round(H - y1)) + pad   # y-up → y-down 像素行
    l, r = int(round(x1)) - pad, int(round(x2)) + pad
    t, b = max(0, t), min(H, max(t + 1, b))
    l, r = max(0, l), min(W, max(l + 1, r))
    a = img[t:b, l:r][::-1]                               # 翻成 y-up（arr[0] = 下 = 低端）
    if a.size == 0 or min(a.shape) < 3:
        return None
    ax = _axis([x1, y1, x2, y2], ctype)
    ys = np.arange(b - t)[::-1] + (H - b)                 # 每行对应的 y-up 坐标
    xs = np.arange(r - l) + l                             # 每列对应的 x 坐标
    if ax == "horiz":                                     # 行索引改成 x：arr[0] = 左 = 低端
        a = a.T
        axis_body, perp_body = (xs >= x1) & (xs <= x2), (ys >= y1) & (ys <= y2)
    else:
        axis_body, perp_body = (ys >= y1) & (ys <= y2), (xs >= x1) & (xs <= x2)
    lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
    if hi - lo < 8:                                       # 几乎纯色 ⇒ 没有标记
        return None
    # ⚠⚠ **分母必须是"本体"**（2026-09-22 修）：`pad` 只用来**看见框外的标记**（LED 箭头），
    #   不能进覆盖率/连续段的归一化分母 —— 否则 pad 一大，横杠相对整幅宽度就必然掉阈值，
    #   实测 pad=10 时 led 的 train 被整类筛没（那是度量缺陷，不是没标记）。
    return a < lo + 0.55 * (hi - lo), axis_body, perp_body


def _run_frac(row: np.ndarray) -> float:
    """该行（垂直于轴）最长**连续暗段**占宽度的比例 —— 用来确认这是一条"整条横线"。"""
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best / max(1, len(row))


def mark_side(dark: np.ndarray, min_strength: float = 0.55, perp_body=None,
              axis_body=None, *, spike: float = 0.15, gap: int = 4, margin: int = 3,
              run_frac: float = 0.6) -> tuple[int, float, float] | None:
    """**尖峰分支**：找"细竖线"（二极管阴极条），返回 `(side, strength)`。

    两步（2026-09-22 用户目视两次订正）：
      1. 沿轴覆盖率 `cov` 找**尖峰**（自身高、离 ±`gap` 处明显低）—— 排除"三角形填充渐变"
         与"两端伸出的轴向残段"（实测：只做这步 d 43.9% ⇒ 61.5%）；
      2. **该位置的暗像素必须是一条连续横段**（最长连续 run ≥ `run_frac × 宽`）——
         排除偶发命中三角形斜边/噪点（v3 新增）。
    ⚠ **不做取反**：方向由 `main` 在 train 上按一致率选（v2 的硬编码 flip 曾把 led 从 88.6% 打到 11.4%）。
    """
    if dark is None or dark.shape[0] < 2 * gap + 3:
        return None
    cols = slice(None) if perp_body is None else perp_body
    cov = dark[:, cols].mean(axis=1)
    n = len(cov)
    # ⚠⚠ **只在"本体内部中段"找候选**（2026-09-22 诊断出的根因）：本体自身的上下边**也是整条
    #   横线**，与阴极条的 prominence 完全打平，`max()` 取第一个 ⇒ **永远是本体边赢**、
    #   预测恒定塌成同一侧（三个不同 bar 位置输出逐字相同 = 铁证）。margin=3px 把边缘剪掉。
    if axis_body is not None:
        idx = np.flatnonzero(axis_body)
        lo_i, hi_i = (int(idx[0]) + margin, int(idx[-1]) - margin) if len(idx) else (0, n - 1)
    else:
        lo_i, hi_i = 0, n - 1
    cands = []
    for i in range(max(0, lo_i), min(n, hi_i + 1)):
        prom = float(cov[i] - max(cov[max(0, i - gap)], cov[min(n - 1, i + gap)]))
        if cov[i] >= min_strength and prom >= spike and _run_frac(dark[i, cols]) >= run_frac:
            cands.append((float(cov[i]), prom, i))
    if not cands:
        return None
    _s, _prom, i = max(cands, key=lambda t: (t[1], t[0]))
    return (-1 if i < n / 2 else +1), float(cov[i]), i / max(1, n - 1)


def broad_peak_side(dark: np.ndarray, min_strength: float = 0.5,
                    perp_body=None) -> tuple[int, float, float] | None:
    """**宽峰分支**（`box`/`fuse` 这类"整排引脚"标记）：不要求尖峰，只看覆盖率最高的位置落在哪一端。

    v2 用尖峰判据把 box 从 59 例筛到 7 例 ⇒ 两类标记**必须分开判**（用户 09-22 定）。
    """
    if dark is None or dark.shape[0] < 6:
        return None
    cov = dark[:, slice(None) if perp_body is None else perp_body].mean(axis=1)
    i = int(np.argmax(cov))
    if float(cov[i]) < min_strength:
        return None
    n = len(cov)
    lo_m = float(np.mean(cov[: max(1, n // 3)]))
    hi_m = float(np.mean(cov[-max(1, n // 3):]))
    return (-1 if lo_m >= hi_m else +1), float(cov[i]), i / max(1, n - 1)


BRANCH = {"box": broad_peak_side, "fuse": broad_peak_side}
# 按类的 pad（用户 09-22 目视 + 实测）：d 的阴极条/led 的箭头要**看见框外**（pad>0）；
# box/fuse 的标记（整排引脚/边缘）本来就在框内，pad 会把周围线拉进来污染宽峰 ⇒ pad=0。
# 2026-09-22 v4 实测：**只有 d 吃 pad**（66.5% vs pad0 61.9%）；led 的 87% 来自**本体横杠**、
# 不吃框外箭头（pad=10 反而 79.8%）⇒ led 回到 0。box/fuse 的标记在框内、怕污染 ⇒ 0。
PAD_BY_TYPE = {"d": 10, "led": 0, "box": 0, "fuse": 0, "crystal_2pin": 0}


def collect(cases: list[tuple[str, Path, Path]], want_types: set[str], *,
            min_strength: float = 0.55, pad: int = 0):
    """→ `(rows, [crop arrays])`；`rows` = (type, src, pin1_is_low, mark_is_low, strength)。"""
    from PIL import Image
    rows, sheet = [], []
    hook_stat: dict = defaultdict(lambda: [0, 0])      # (type, src, has_hook) -> [agree, n]
    for src, tfile, png in cases:
        t = json.loads(tfile.read_text(encoding="utf-8"))
        with Image.open(png) as im:
            img = np.asarray(im.convert("L"))
        for k, c in (t.get("components") or {}).items():
            ty = c.get("type", "?")
            if ty not in want_types:
                continue
            pd = (t.get("pins") or {}).get(k) or {}
            if len(pd) != 2 or "pin_1" not in pd:
                continue
            p1 = pd["pin_1"].get("point") or {}
            if "x" not in p1:
                continue
            x1, y1, x2, y2 = [float(v) for v in c["bbox"]]
            ax = _axis([x1, y1, x2, y2], ty)
            t1 = ((float(p1["y"]) - y1) / max(1e-6, y2 - y1) if ax == "vert"
                  else (float(p1["x"]) - x1) / max(1e-6, x2 - x1))
            p_ty = pad if pad >= 0 else PAD_BY_TYPE.get(ty, 0)   # pad<0 = 按类自动
            prof = axis_profile(img, [x1, y1, x2, y2], ty, p_ty)
            if prof is None:
                continue
            dark, axis_body, perp_body = prof
            fn = BRANCH.get(ty, mark_side)            # box/fuse 走宽峰；其余走尖峰
            r = (fn(dark, min_strength, perp_body, axis_body) if fn is mark_side
                 else fn(dark, min_strength, perp_body))
            if r is None:
                continue
            side, s, i_frac = r
            # **二级判据（d 用）**：在尖峰那端看"横杠的哪一侧有更多墨迹"——二极管是
            # 三角+横杠，**三角填充在横杠的一侧**；取两侧相邻带的平均覆盖率之差定侧。
            n_row = dark.shape[0]
            i = int(round(i_frac * (n_row - 1)))
            band, cols = 8, perp_body
            prev = float(dark[max(0, i - band):max(1, i - 1), cols].mean()) if i > 2 else 0.0
            nxt = (float(dark[min(n_row - 1, i + 2):min(n_row, i + 1 + band), cols].mean())
                   if i < n_row - 3 else 0.0)
            rows.append((ty, src, t1 < 0.5, side < 0, prev >= nxt, s))
            # **带钩判据（Schottky/Zener）**：横杠自身的列跨度之外、紧邻的行里还有暗像素。
            row_cols = np.flatnonzero(dark[i, cols])
            has_hook = False
            if len(row_cols) >= 2:
                c0, c1 = int(row_cols[0]), int(row_cols[-1])
                for dj in (-2, -1, 1, 2):
                    j = i + dj
                    if not (0 <= j < n_row):
                        continue
                    outside = dark[j, cols].copy()
                    outside[c0:c1 + 1] = False
                    if outside.any():
                        has_hook = True
                        break
            hs = hook_stat[(ty, src, has_hook)]
            hs[0] += int((t1 < 0.5) == (side < 0))
            hs[1] += 1
            if src == "10gt" or (src == "train" and len(sheet) < 12):
                H = img.shape[0]
                sheet.append((img[max(0, int(H - y2) - p_ty):int(H - y1) + p_ty or 1,
                                  max(0, int(x1) - p_ty):int(x2) + p_ty or 1],
                              ty, src, t1 < 0.5, side < 0, ax, i_frac))
    return rows, sheet, hook_stat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="probe_polarity_marks", description="极性标记（跨工具）")
    ap.add_argument("--types", default="d,led,box,mosfet,bjt,fuse,crystal_2pin")
    ap.add_argument("--min-strength", type=float, default=0.55)
    ap.add_argument("--pad", type=int, default=-1,
                    help="符号框外多留的像素；**-1 = 按类自动**（d/led=10，box/fuse=0；"
                         "依据：d 阴极条与 led 箭头需要看见框外，box 怕被周围线污染）")
    ap.add_argument("--sheet", default=None, help="目视拼图输出路径（png）")
    a = ap.parse_args(argv)
    want = {x.strip() for x in a.types.split(",") if x.strip()}

    ds, tg, gt = ROOT / "Dataset300", ROOT / "赛题六公开数据集" / "200_train_cases", ROOT / "10gt" / "10gt"
    cases: list[tuple[str, Path, Path]] = []
    for cid in sorted(d.name for d in tg.iterdir() if d.is_dir()):
        f = next(iter(sorted(tg.joinpath(cid).glob("*_target_named.json"))), None)
        p = next(iter(sorted(ds.joinpath(cid).glob("*.png"))), None) if (ds / cid).is_dir() else None
        if f and p:
            cases.append(("train", f, p))
    for d in sorted(gt.iterdir()):
        if d.is_dir():
            f, p = next(iter(d.glob("*.json")), None), next(iter(d.glob("*.png")), None)
            if f and p:
                cases.append(("10gt", f, p))
    rows, sheet, hook_stat = collect(cases, want, min_strength=a.min_strength, pad=a.pad)
    print(f"有横杠标记的样本 {len(rows)}（总 2 脚元件里按 min_strength={a.min_strength} 筛出）")
    agg = defaultdict(list)
    for ty, src, p1low, mk_low, tri_low, _s in rows:
        agg[(ty, src)].append((p1low, mk_low, tri_low))
    # ⚠⚠ **方向在 train 上选，不许硬编码**（v2 硬编码 flip 把 led 从 88.6% 打到 11.4% 的教训）
    sign: dict[str, int] = {}
    for ty in {t for t, _ in agg}:
        v = agg.get((ty, "train")) or []
        if len(v) < 5:
            sign[ty] = 1
            continue
        raw = float(np.mean([p == m for p, m, _t in v]))
        sign[ty] = 1 if raw >= 0.5 else -1
    print(f"\n{'type':14s} {'源':6s} {'n':>4s} | {'标记低端':>8s} | {'pin_1低端':>9s} | "
          f"{'一致率(选向后)':>13s} | 选中方向")
    for (ty, src) in sorted(agg, key=lambda z: (z[0], z[1])):
        v = agg[(ty, src)]
        n = len(v)
        if n < 2:
            continue
        sg = sign.get(ty, 1)
        mk = float(np.mean([(m if sg > 0 else (not m)) for _p, m, _t in v]))
        p1 = float(np.mean([p for p, _m, _t in v]))
        agree = float(np.mean([(p == m) if sg > 0 else (p != m) for p, m, _t in v]))
        print(f"{ty:14s} {src:6s} {n:4d} | {mk:8.1%} | {p1:9.1%} | {agree:13.1%} | "
              f"{'原向' if sg > 0 else '取反'}")
    print("\n读法：**方向由 train 选**（每类两方向取一致率高者）；跨工具可迁 ⇔ 两行『一致率』接近。")
    # 二级判据（三角填充侧）单独一张表：它的方向同样在 train 上选
    print(f"\n=== 二级判据：**三角填充侧**（只对有尖峰的类型有意义）===")
    print(f"{'type':14s} {'源':6s} {'n':>4s} | {'三角侧=低端':>10s} | {'一致率(选向后)':>13s}")
    for (ty, src) in sorted(agg, key=lambda z: (z[0], z[1])):
        v = agg[(ty, src)]
        n = len(v)
        if n < 5:
            continue
        vt = agg.get((ty, "train")) or []
        raw = float(np.mean([p == t for p, _m, t in vt])) if len(vt) >= 5 else 0.5
        sg = 1 if raw >= 0.5 else -1
        tri = float(np.mean([(t if sg > 0 else (not t)) for _p, _m, t in v]))
        ag2 = float(np.mean([(p == t) if sg > 0 else (p != t) for p, _m, t in v]))
        print(f"{ty:14s} {src:6s} {n:4d} | {tri:10.1%} | {ag2:13.1%}")
    print(f"\n=== 带钩判据（Schottky/Zener）分组：横杠邻行在其跨度之外还有暗像素 = 带钩 ===")
    print(f"{'type':14s} {'源':6s} {'带钩':>4s} {'n':>4s} | {'一致率(原向)':>12s}")
    for (ty, src, hk) in sorted(hook_stat, key=lambda z: (z[0], z[1], z[2])):
        ok, n = hook_stat[(ty, src, hk)]
        if n < 3:
            continue
        print(f"{ty:14s} {src:6s} {('是' if hk else '否'):>4s} {n:4d} | {ok / n:12.1%}")
    if a.sheet:
        # 拼图带**标注**：🔴 红线 = 探针判出的标记位置；🟢 绿线 = pin_1 真实那一端。
        # 目视只要看"红线是否落在标记上、以及红绿是否同侧"——这比看数字快得多。
        from PIL import Image, ImageDraw
        tw, th, bar = 108, 108, 14
        cols = min(6, max(1, len(sheet)))
        rows_n = int(np.ceil(len(sheet) / cols))
        canvas = Image.new("RGB", (cols * tw, rows_n * (th + bar)), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for i, (crop, ty, src, p1low, mklow, ax, i_frac) in enumerate(sheet):
            if crop.size == 0:
                continue
            ox, oy = (i % cols) * tw, (i // cols) * (th + bar)
            canvas.paste(Image.fromarray(crop).convert("RGB").resize((tw, th)), (ox, oy))
            if ax == "vert":
                y = oy + int((1.0 - i_frac) * (th - 1))          # i_frac=0（低端）→ 图底
                dr.line([(ox, y), (ox + tw - 1, y)], fill=(255, 0, 0), width=2)
                dr.line([(ox, oy + th - 1 if p1low else oy), (ox + tw - 1, oy + th - 1 if p1low else oy)],
                        fill=(0, 160, 0), width=1)
            else:
                x = ox + int(i_frac * (tw - 1))
                dr.line([(x, oy), (x, oy + th - 1)], fill=(255, 0, 0), width=2)
                dr.line([(ox if p1low else ox + tw - 1, oy), (ox if p1low else ox + tw - 1, oy + th - 1)],
                        fill=(0, 160, 0), width=1)
            dr.text((ox + 2, oy + th + 1), f"{ty}/{src} pL{int(p1low)} mL{int(mklow)}",
                    fill=(0, 0, 0))
            print(f"  [{i}] {ty}/{src} pin_1_low={int(p1low)} mark_low={int(mklow)} axis={ax}")
        canvas.save(a.sheet)
        print(f"[probe_polarity_marks] 标注拼图 -> {a.sheet}（🔴=探针判的标记位置，🟢=pin_1 端）")
    return 0


def source_n(agg, ty) -> int:
    return sum(len(v) for (t, _s), v in agg.items() if t == ty)


if __name__ == "__main__":
    sys.exit(main())
