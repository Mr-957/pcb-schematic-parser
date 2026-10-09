# -*- coding: utf-8 -*-
"""P1 阶段 0.5.3：引脚几何推导（**永久模块，不是兜底**；PinF1 权重 0.25）。

**原理（实测，非推断）**：官方 `bbox` 是**符号体、不含引脚**，引脚点落在框外**恒定偏移**处。
对 `r/c/d/led/l/fuse/crystal_2pin` 实测：
  - 垂直于引脚轴的坐标**恒等于 bbox 中心**（垂直残差 100% 为 0）
  - 沿轴方向的偏移是**每类一个常数**（`fit_offsets()` 可从数据重新导出，不是魔数）
  - 2-pin 的 `pin_1` 方向**按类型查表**（见 `PIN1_SIDE`）—— 聚合值会骗人，
    `d` 基本是**反的**（竖直 25.6% / 水平 31.6%），`led` 的水平方向也是反的（33.3%）

⚠ **`crystal_2pin` 的长宽比与引脚轴是反的**（用 `h>=w→竖直` 判它，一致率 **0%**）→ 特例。

**⚠⚠ 本模块的成败取决于框精度**：引脚坐标**整份继承框边缘误差（1:1）**，而引脚容差只有
**5px**、元件容差是 **20px**。若检测框中位误差 ~8px（"对 compF1 够用"的水平），
本模块**直接作废**（key 免费但 point 全错）。因此 `measure_ceiling()` 必须连同
**jitter 敏感性曲线**一起跑 —— **这条曲线要在花任何训练预算之前先看**。

⚠ `pinname` 是硬约束：`scorer.pin_score` 要求 `pinname` 与 pin key **同时**正确。
实测把 `pinname` 置空会让 pinF1 **从 0.4505 直接掉到 0**。本模块只产出坐标与 key，
`pinname` 由上层（OCR/编号）补齐，**不可留空**。

坐标：官方 target 为 **y-up 原点左下**；本模块全程 y-up，不做任何翻帧。

运行（pcbeval env，项目根）：
  python -m pcbmodel.integrate.pin_geom --ceiling        # 孤立上限 + jitter 曲线
  python -m pcbmodel.integrate.pin_geom --fit            # 从数据重导偏移表
  python -m pcbmodel.integrate.pin_geom --case 0025      # 单例推导明细
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.config import EvalConfig, OFFICIAL_TRAIN_ROOT
from pcbeval.caseio import discover_official
from pcbeval.scorer import pin_score

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "stage05"
VOID = "∅"
POINT_TOL = 5.0

# --- 偏移表：引脚点在 bbox 外多远（px）。None = 不恒定，必须交给检测。---------
PIN_OFFSET_PX: dict[str, float | None] = {
    "r": 10.0, "c": 10.0, "d": 10.0, "led": 10.0, "fuse": 10.0,
    "mosfet": 10.0, "bjt": 10.0, "mosfet_npn": 10.0, "mosfet_pnp": 10.0,
    "bjt_npn": 10.0, "bjt_pnp": 10.0, "crystal_4pin": 10.0,
    "l": 3.0, "switch": 3.0, "amp": 3.0,
    "pin": 20.0, "testpoint": 20.0, "amp_3pin": 20.0,
    "crystal_2pin": 15.0, "crystal_3pin": 15.0,
    "box": None,          # 中位 10 / p90 19 —— 不恒定，需检测
    "amp_5pin": None,     # p10 10 / 中位 20 —— 不恒定
}

# 长宽比 → 引脚轴 对本类**不成立**（实测一致率 0%）→ 取反
PIN_AXIS_INVERTED = frozenset({"crystal_2pin"})

# pin_1 落在哪一侧："low" = 竖直时在下 / 水平时在左。
# ⚠ 这是**每类一张表**，不是全局规则：`d` 基本是反的、`led` 水平方向也是反的。
PIN1_SIDE: dict[str, str] = {
    "r": "low", "c": "low", "l": "low", "fuse": "low", "crystal_2pin": "low",
    "switch": "low", "pin": "low", "testpoint": "low",
    "led": "ambig", "d": "ambig", "box": "ambig",
}
# 只有这些类可以无条件用 pin_1=low 推导（实测接近 100%）
PIN1_TRUSTWORTHY = frozenset({"r", "c", "l", "fuse", "crystal_2pin"})

DERIVABLE_2PIN = frozenset(k for k, v in PIN_OFFSET_PX.items() if v is not None)

# --- 1 脚符号（电源/地/网络端口）的连接点 ------------------------------------ #
# 实测（200_train + 10gt，3616 个 1 脚符号，2026-09-22）：
#   - 锚点 = bbox 边外 **恒定 10px**（p10 = p90 = 10.0，出奇地干净）；
#     垂直于该边的坐标 = bbox 中心（gnd 79% / v 78% 满足 |dx| < 2px）。
#   - ⚠⚠ **边随朝向变**：gnd 上 72% / 左 11% / 右 10% / 下 7%；v 下 73% / 右 12% / 左 11%；
#     net_input 右 42% / 左 41%。全量 3616 个里有 **24.4% 不在"多数朝向"那侧**
#     ⇒ 固定边只能对 ~76%，错的 24.4% 会把锚点放到对面 20px 处（> 5px 容差）⇒ 既挂不上
#     网、又白送一个 FP。故提供 wire-facing：给一组导线点就取**最近的候选锚点**。
#   - `net_short` **不在本表**：它外向只有 1px，且 149/162 是 **2 脚** ⇒ 属 2 脚表（见
#     `PIN_OFFSET_PX` 的旁注与 MODEL_HANDOFF #76 的 FP 代价说明）。
SYMBOL_ANCHOR_PX: dict[str, float] = {
    "gnd": 10.0, "v": 10.0,
    "net_input": 10.0, "net_output": 10.0, "net_bidirection": 10.0,
}
# 没有导线信息时的兜底边（多数朝向）：gnd 的脚朝上（y-up 里 y2 之外），v 朝下。
SYMBOL_SIDE_PRIOR: dict[str, str] = {"gnd": "top", "v": "bottom"}
ANCHOR_SIDES = ("top", "bottom", "left", "right")


def _d2(p, q) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def anchor_candidates(bbox, ctype: str) -> list[tuple[str, tuple[float, float]]]:
    """该符号的 4 个候选连接点 `(side, point)`，均为 bbox 边外 `SYMBOL_ANCHOR_PX` px。"""
    off = SYMBOL_ANCHOR_PX.get(ctype)
    if off is None:
        return []
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [("top", (cx, y2 + off)), ("bottom", (cx, y1 - off)),
            ("left", (x1 - off, cy)), ("right", (x2 + off, cy))]


def derive_symbol_anchor(bbox, ctype: str, wire_pts=None, *, tol: float | None = None):
    """1 脚符号的连接点 → `(point, side)`；类型不在表里返回 None。

    `wire_pts` = 导线点/段端点的可迭代 `(x, y)`（y-up，与 bbox 同帧）：
      - 给了就用 **wire-facing**：取离它最近的候选锚点；
      - 最近的那个仍超出 `tol` ⇒ 返回 None（该符号附近压根没导线，**不造点**，避免多一个孤立 FP）；
      - 没给就用 `SYMBOL_SIDE_PRIOR` 兜底（只有 gnd/v 有先验，其余类型返回 None）。

    ⚠⚠ `tol` 是**选边的搜索半径**（默认 `2 × offset` = 20px），**不是挂网容差** ——
    能否挂上仍由 `ceiling.cluster_components` 的 5px 决定。两者混用会出错：导线从左侧进
    时它的端点就在候选点附近 ±2px，但若导线画到符号体边缘就停（中位端点差 2px、p90 更大），
    用 5px 会把**正确的边**判掉（2026-09-22 自检抓到过）。
    """
    cands = anchor_candidates(bbox, ctype)
    if not cands:
        return None
    pts = [(float(p[0]), float(p[1])) for p in (wire_pts or [])]
    if pts:
        side, pt = min(cands, key=lambda c: min(_d2(c[1], q) for q in pts))
        r = (2.0 * SYMBOL_ANCHOR_PX[ctype]) if tol is None else float(tol)
        return (pt, side) if min(_d2(pt, q) for q in pts) <= r * r else None
    side = SYMBOL_SIDE_PRIOR.get(ctype)
    if side is None:
        return None
    return next(((pt, s) for s, pt in cands if s == side), None)


def selftest() -> int:
    """锚点几何自检（纯几何，不依赖数据集/torch）。"""
    bb = [100.0, 200.0, 119.0, 220.0]          # 19 x 20，y-up
    assert anchor_candidates(bb, "r") == [], "2 脚类型不该有 1 脚锚点"
    cands = dict(anchor_candidates(bb, "gnd"))
    assert cands["top"] == (109.5, 230.0), cands          # cx, y2 + 10
    assert cands["bottom"] == (109.5, 190.0), cands
    assert cands["left"] == (90.0, 210.0), cands
    assert cands["right"] == (129.0, 210.0), cands

    # 兜底：gnd 朝上、v 朝下
    assert derive_symbol_anchor(bb, "gnd")[1] == "top"
    assert derive_symbol_anchor(bb, "v")[1] == "bottom"
    assert derive_symbol_anchor(bb, "net_input") is None   # 无先验、无导线 → 不猜

    # wire-facing：导线在左侧 ⇒ 选左；在下方 ⇒ 选下（24.4% 翻转样本的那一类）
    assert derive_symbol_anchor(bb, "gnd", [(80.0, 210.0)]) == ((90.0, 210.0), "left")
    assert derive_symbol_anchor(bb, "gnd", [(109.5, 188.0)]) == ((109.5, 190.0), "bottom")
    # 最近的导线也超出 tol ⇒ None（不造孤立点）
    assert derive_symbol_anchor(bb, "gnd", [(60.0, 210.0)], tol=5.0) is None
    assert derive_symbol_anchor(bb, "gnd", [(60.0, 210.0)], tol=35.0) == ((90.0, 210.0), "left")
    print("[pin_geom] 1 脚锚点 selftest OK（10px / 4 候选 / wire-facing / tol 闸）")
    return 0


def pin_axis(bbox, ctype: str) -> str:
    """返回 `"vert"` 或 `"horiz"`。判据 = bbox 长宽比（`crystal_2pin` 取反）。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    vert = (y2 - y1) >= (x2 - x1)
    if ctype in PIN_AXIS_INVERTED:
        vert = not vert
    return "vert" if vert else "horiz"


def derive_2pin(bbox, ctype: str, *, pin1_side: str | None = None
                ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """2 脚元件的 `(pin_1 点, pin_2 点)`。类型不可推导时返回 None。"""
    off = PIN_OFFSET_PX.get(ctype)
    if off is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    lo_side = pin1_side if pin1_side is not None else PIN1_SIDE.get(ctype, "low")
    if pin_axis(bbox, ctype) == "vert":
        a, b = (cx, y1 - off), (cx, y2 + off)          # a = 低端（下）
    else:
        a, b = (x1 - off, cy), (x2 + off, cy)          # a = 低端（左）
    if lo_side == "high":
        a, b = b, a
    return a, b


def derive_pins(bbox, ctype: str, n_pins: int, *, pin1_side: str | None = None
                ) -> list[tuple[float, float]] | None:
    """通用入口。只有 2 脚且类型可推导时给结果；否则 None（交给检测）。"""
    if n_pins != 2 or ctype not in DERIVABLE_2PIN:
        return None
    got = derive_2pin(bbox, ctype, pin1_side=pin1_side)
    return list(got) if got else None


def bbox_from_pin(pin_pt, ctype: str, axis: str, *, pin1_is_low: bool = True):
    """**逆向**：由引脚点反推 bbox —— 给阶段 2（U-Net 导线端点即引脚尖）闭环用。

    `offset` 可推导类才可用；`axis` 给 `"vert"`/`"horiz"`。返回 `[x1,y1,x2,y2]`。
    """
    off = PIN_OFFSET_PX.get(ctype)
    if off is None:
        return None
    px, py = float(pin_pt[0]), float(pin_pt[1])
    # 只给一个引脚点无法定出框长，调用方须传另一端的对称信息；此处按"该点为低端"给出半框
    if axis == "vert":
        return [px, py + off, px, py + off]            # 退化，仅占位（阶段 2 用两端点）
    return [px + off, py, px + off, py]


# --------------------------------------------------------------------------- #
# 数据侧：从官方 target 取样
# --------------------------------------------------------------------------- #
def iter_train_targets(limit: int = 0):
    ids = sorted(d.name for d in Path(OFFICIAL_TRAIN_ROOT).iterdir() if d.is_dir())
    for i, cid in enumerate(ids):
        if limit and i >= limit:
            break
        oc = discover_official(cid, OFFICIAL_TRAIN_ROOT, kind="train")
        if oc.target is None:
            continue
        with open(oc.target, encoding="utf-8") as f:
            yield cid, json.load(f)


def fit_offsets(limit: int = 0) -> dict:
    """从数据**重新导出**偏移表（每类取众数），使表不脱离数据。"""
    from collections import Counter
    tallies: dict[str, Counter] = {}
    for _cid, t in iter_train_targets(limit):
        comps, pins = t.get("components", {}), t.get("pins", {})
        for d, c in comps.items():
            if d.startswith(VOID):
                continue
            pd = pins.get(d) or {}
            if len(pd) != 2:
                continue
            ty = c.get("type")
            x1, y1, x2, y2 = (float(v) for v in c["bbox"])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            axis = pin_axis(c["bbox"], ty)
            seen = []
            for p in pd.values():
                pt = p.get("point") or {}
                if "x" not in pt:
                    continue
                px, py = float(pt["x"]), float(pt["y"])
                if axis == "vert":
                    # 垂直残差应为 0；沿轴偏移 = 到最近水平边的距离
                    if abs(px - cx) <= 1.0:
                        seen.append(min(abs(py - y1), abs(py - y2)))
                else:
                    if abs(py - cy) <= 1.0:
                        seen.append(min(abs(px - x1), abs(px - x2)))
            if len(seen) == 2:
                tallies.setdefault(ty, Counter())[round(max(seen), 1)] += 1
    fitted = {ty: (ct.most_common(1)[0][0] if ct else None)
              for ty, ct in sorted(tallies.items())}
    return fitted


# --------------------------------------------------------------------------- #
# 上限测量 + jitter 曲线
# --------------------------------------------------------------------------- #
def _pred_with_derived(t: dict, *, jitter: float, rng: random.Random,
                       pinname_from_gt: bool = True, only_types=None) -> dict:
    """用 GT 框（可加 jitter）推导引脚，其余照抄 GT。返回可喂给 pin_score 的 pred。"""
    comps, gt_pins = t.get("components", {}), t.get("pins", {})
    pc, pp = {}, {}
    for d, c in comps.items():
        pc[d] = dict(c)
        pd = gt_pins.get(d) or {}
        ty = c.get("type")
        if d.startswith(VOID) or len(pd) != 2 or ty not in DERIVABLE_2PIN or \
                (only_types is not None and ty not in only_types):
            pp[d] = {}
            continue
        bb = [float(v) for v in c["bbox"]]
        if jitter:
            bb = [v + rng.uniform(-jitter, jitter) for v in bb]
            bb = [bb[0], bb[1], max(bb[0] + 0.5, bb[2]), max(bb[1] + 0.5, bb[3])]
        got = derive_2pin(bb, ty)
        if got is None:
            pp[d] = {}
            continue
        names = {}
        if pinname_from_gt:
            for k, p in pd.items():
                names[k] = p.get("pinname")
        pp[d] = {"pin_1": {"pinname": names.get("pin_1"), "point": {"x": got[0][0], "y": got[0][1]}},
                 "pin_2": {"pinname": names.get("pin_2"), "point": {"x": got[1][0], "y": got[1][1]}}}
    return {"components": pc, "pins": pp, "nets": {}}


def _exact_point_rate(t: dict, *, jitter: float, rng: random.Random) -> tuple[int, int]:
    """推导点与 GT 点的 ≤5px 命中率（只看可推导的 2 脚元件）。"""
    hit = tot = 0
    comps, gt_pins = t.get("components", {}), t.get("pins", {})
    for d, c in comps.items():
        if d.startswith(VOID):
            continue
        pd = gt_pins.get(d) or {}
        ty = c.get("type")
        if len(pd) != 2 or ty not in DERIVABLE_2PIN:
            continue
        bb = [float(v) for v in c["bbox"]]
        if jitter:
            bb = [v + rng.uniform(-jitter, jitter) for v in bb]
            bb = [bb[0], bb[1], max(bb[0] + 0.5, bb[2]), max(bb[1] + 0.5, bb[3])]
        got = derive_2pin(bb, ty)
        if got is None:
            continue
        for k, pt in zip(("pin_1", "pin_2"), got):
            g = (pd.get(k) or {}).get("point")
            if not g or "x" not in g:
                continue
            tot += 1
            if abs(g["x"] - pt[0]) <= POINT_TOL and abs(g["y"] - pt[1]) <= POINT_TOL:
                hit += 1
    return hit, tot


def measure_ceiling(limit: int = 0, jitters=(0.0, 2.0, 5.0, 10.0),
                    cfg: EvalConfig | None = None, seed: int = 0) -> dict:
    """GT 框输入下的孤立上限 + jitter 敏感性曲线。"""
    cfg = cfg or EvalConfig()
    cases = list(iter_train_targets(limit))
    rows = []
    base = None
    for j in jitters:
        rng = random.Random(seed)
        TP = FP = FN = 0
        hit = tot = 0
        for _cid, t in cases:
            h, n = _exact_point_rate(t, jitter=j, rng=rng)
            hit += h
            tot += n
            r = pin_score(_pred_with_derived(t, jitter=j, rng=rng), t, cfg)
            TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
        row = {"jitter_px": j, "point_exact_rate": round(hit / tot, 4) if tot else 0.0,
               "n_points": tot, "pin_f1": round(f1, 4),
               "tp": TP, "fp": FP, "fn": FN}
        rows.append(row)
        if j == 0.0:
            base = row
    # n_pins 置空的对照（证明 pinname 是硬约束）
    rng = random.Random(seed)
    TP = FP = FN = 0
    for _cid, t in cases:
        r = pin_score(_pred_with_derived(t, jitter=0.0, rng=rng,
                                         pinname_from_gt=False), t, cfg)
        TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
    f1_noname = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    return {"n_cases": len(cases), "rows": rows, "baseline": base,
            "pin_f1_without_pinname": round(f1_noname, 4),
            "derivable_types": sorted(DERIVABLE_2PIN)}


def render_md(res: dict) -> str:
    L = ["# 引脚几何推导：孤立上限 + jitter 敏感性曲线", "",
         f"> 输入 = **GT 框**，源 = `赛题六公开训练集`（{res['n_cases']} 例）", "",
         "## jitter 曲线（**花训练预算前先看这张表**）", "",
         "| 框 jitter (px) | 推导点 ≤5px 命中率 | pinF1 | tp | fp | fn |",
         "|---:|---:|---:|---:|---:|---:|"]
    for r in res["rows"]:
        L.append(f"| ±{r['jitter_px']:.0f} | {r['point_exact_rate'] * 100:.2f}% | "
                 f"**{r['pin_f1']:.4f}** | {r['tp']} | {r['fp']} | {r['fn']} |")
    b = res["baseline"] or {}
    L += ["", "## 读法", "",
          f"- **框误差 0 时 pinF1 = {b.get('pin_f1', 0):.4f}**，这是本模块的隔离上限。",
          f"- 引脚容差只有 **5px**、而元件容差是 **20px**：曲线若在 ±2~±5 之间掉崖，"
          f"说明检测框精度直接决定本模块存亡。",
          f"- ⚠ **`pinname` 置空的对照：pinF1 = {res['pin_f1_without_pinname']:.4f}**"
          f" —— 证明 `pinname` 与 pin key 必须**同时**正确，几何再准也没用。",
          f"- 覆盖类型：`{'`, `'.join(res['derivable_types'])}`；"
          f"`box`/`amp_5pin` 不在内（偏移不恒定），须检测。", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="pin_geom", description="引脚几何推导 / 上限测量")
    ap.add_argument("--ceiling", action="store_true", help="跑孤立上限 + jitter 曲线")
    ap.add_argument("--fit", action="store_true", help="从数据重导偏移表")
    ap.add_argument("--selftest", action="store_true",
                    help="1 脚符号锚点几何自检（不依赖数据集/torch）")
    ap.add_argument("--case", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jitters", default="0,2,5,10")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.fit:
        fitted = fit_offsets(a.limit)
        print("[pin_geom] 从数据重导的偏移表（众数）：")
        drift = []
        for ty, v in fitted.items():
            cur = PIN_OFFSET_PX.get(ty, "<<不在表里>>")
            mark = "" if cur == v else f"   ⚠ 表里是 {cur}"
            if cur != v:
                drift.append(ty)
            print(f"   {ty:16s} {v}{mark}")
        print(f"[pin_geom] 与硬编码表不一致的类：{drift or '无'}")
        return 0

    if a.case:
        for cid, t in iter_train_targets(0):
            if cid != a.case:
                continue
            print(f"[pin_geom] {cid}")
            for d, c in t["components"].items():
                if d.startswith(VOID):
                    continue
                ty, bb = c["type"], c["bbox"]
                got = derive_pins(bb, ty, len(t.get("pins", {}).get(d) or {}))
                gt = t.get("pins", {}).get(d) or {}
                gs = {k: (v["point"]["x"], v["point"]["y"]) for k, v in gt.items()}
                if got:
                    print(f"   {d:8s} {ty:14s} 推导 pin_1={got[0]} pin_2={got[1]}"
                          f"   GT={gs}")
                else:
                    print(f"   {d:8s} {ty:14s} 不可推导（交检测）   GT={gs}")
            return 0
        print(f"[pin_geom] 找不到 case {a.case}")
        return 1

    if a.ceiling:
        jit = tuple(float(x) for x in a.jitters.split(",") if x.strip())
        res = measure_ceiling(a.limit, jit)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pin_geom_ceiling.md").write_text(render_md(res), encoding="utf-8")
        (out / "pin_geom_ceiling.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        b = res["baseline"] or {}
        print(f"[pin_geom] {res['n_cases']} 例 ｜ 框误差 0 时 pinF1 = "
              f"{b.get('pin_f1', 0):.4f}（点命中 {b.get('point_exact_rate', 0) * 100:.2f}%）")
        for r in res["rows"][1:]:
            print(f"   jitter ±{r['jitter_px']:.0f}px -> pinF1 {r['pin_f1']:.4f}"
                  f"（点命中 {r['point_exact_rate'] * 100:.2f}%）")
        print(f"   ⚠ pinname 置空对照 = {res['pin_f1_without_pinname']:.4f}")
        print(f"   -> {out / 'pin_geom_ceiling.md'}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
