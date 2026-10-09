# -*- coding: utf-8 -*-
"""box（IC）引脚链：**导线附着点 + 体侧脚名文本 + 官方库反查** -> `DraftPin`。

     wire_pts（骨架段端点）
        │ ① 距元件框 <= edge_tol 且不在框内 -> 候选脚点（按最近面记）
        ▼
     候选点 ── ② 轴向规则（到 "点→框内" 射线段 <= r）+ **元件内 1:1 唯一** ——> 脚名串
        │ ③ `pin_from_name.number_for`（库唯一名 -> 投票表 -> 弃权）
        ▼
     DraftPin(point, pin_number_label, pinname)

## 上游证据（2026-09-23 两个探针，别再重复量）

* 跳 0 `probes/box_chain_ceiling.py`（HANDOFF 89，纯 GT）：box 脚 **100% 在元件框外**、
  中位 10px（p90 19）；`axis` 规则 **覆盖内 94.5%**、全脚口径 71.4%；**型号精确对库覆盖
  74.7% 的 box 脚**；由脚名文本反推脚点只有 51%（腿长 p10 4 / p90 22 不恒定）
  ⇒ **脚点必须来自导线**（`76`：89.6% 的 box GT 脚落在 GT 导线端点 5px 内）。
* 跳 0.5 `probes/box_recall.py`（HANDOFF 90，真管线）：检出 94.7% / 识别 85.1% /
  端到端可用 80.6%；⚠ **候选不许按 `cls=='pinName'` 过滤**（纯数字脚名类纯度只有 3.5%）。

## 三条硬约定

1. **宁可弃权不许猜** —— 写错一个脚 = FP+FN 双扣，少写一个只扣 FN。
2. **绝不碰 `pin_geom.PIN_OFFSET_PX`**（`76`：`derive_pins` 恒返回 2 点 ⇒ 给 box 加表项
   会让每个元件白送 FP）。本模块只被显式调用，不进那张表。
3. **只填"仍然 0 脚"的 box**（不覆盖检测/几何/1 脚锚点已给出的脚）。

⚠ **投票表来自官方 train GT 的汇总**（`ic_db.build`，性质与训练权重同源）；
`use_vote=False` 可退化成"只用官方库"，做消融。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from pcbmodel.integrate import pin_from_name
from pcbmodel.integrate.numbering import DraftPin

# y-up 框内法向 —— **与 `probes/box_chain_ceiling.NORM_IN` 同一张表，改一处要同步另一处**
NORM_IN = {"L": (1.0, 0.0), "R": (-1.0, 0.0), "B": (0.0, 1.0), "T": (0.0, -1.0)}

BOX_EDGE_TOL = 20.0    # 候选脚点：距元件框 <= 该值（跳 0：框外中位 10 / p90 19）
BOX_TEXT_R = 15.0      # 脚名候选：到"点->框内"射线段 <= 该值（跳 0：r10~r20）
POINT_MIN_SEP = 3.0    # 候选点去重（同一条线的两端/骨架拐点会贴在一起）
INSIDE_TOL = 2.0       # 允许略微落在框内（矢量化误差）
MODEL_MIN_LEN = 4      # 型号串最小长度（库里混着 `B`/`D`/`U1`/`OUT` 这类短噪声键）
MODEL_R = 100.0        # 型号文本中心到元件框的距离上限（实测真型号多在 15~80px）
# 位号/短噪声形态：`U1`/`D3`/`R12` 这类**不许**当型号（库里有同名键）
_JUNK_KEY = re.compile(r"^[A-Z]{1,3}\d{0,3}$")


# --------------------------------------------------------------------------- #
# 几何（与探针逐字同定义，别再写第二套）
# --------------------------------------------------------------------------- #
def face_of(p, b) -> tuple[str, float, float]:
    """点相对框：`(最近面, 有符号距离, 到该面的轴向距离)`；框内为负、框外为正。"""
    x1, y1, x2, y2 = b
    px, py = p
    if x1 < px < x2 and y1 < py < y2:
        d = {"L": px - x1, "R": x2 - px, "B": py - y1, "T": y2 - py}
        f = min(d, key=d.get)
        return f, -d[f], d[f]
    qx = min(max(px, x1), x2)
    qy = min(max(py, y1), y2)
    dist = math.hypot(px - qx, py - qy)
    if dist == 0.0:
        d = {"L": abs(px - x1), "R": abs(x2 - px), "B": abs(py - y1), "T": abs(y2 - py)}
        f = min(d, key=d.get)
        return f, 0.0, 0.0
    cands = []
    if px < x1:
        cands.append(("L", x1 - px))
    elif px > x2:
        cands.append(("R", px - x2))
    if py < y1:
        cands.append(("B", y1 - py))
    elif py > y2:
        cands.append(("T", py - y2))
    return max(cands, key=lambda t: t[1])[0], dist, 0.0


def seg_dist(p, a, b) -> tuple[float, float]:
    """点到线段的距离 + 参数 t∈[0,1]。"""
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy)), t


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _pairs(texts):
    """`OcrText`（`.content`/`.bbox`）或 `(content, bbox)` 元组 -> `[(content, bbox)]`。"""
    out = []
    for t in texts or ():
        if hasattr(t, "content") and hasattr(t, "bbox"):
            c, bb = t.content, t.bbox
        else:
            c, bb = t[0], t[1]
        if c is None or not str(c).strip() or bb is None:
            continue
        out.append((str(c), [float(v) for v in bb]))
    return out


# --------------------------------------------------------------------------- #
# ① 候选脚点
# --------------------------------------------------------------------------- #
def edge_points(bbox, wire_pts, *, edge_tol: float = BOX_EDGE_TOL,
                min_sep: float = POINT_MIN_SEP, inside_tol: float = INSIDE_TOL):
    """导线端点里属于这个元件的：**不在框内（容 `inside_tol` 误差）、距框 <= edge_tol**。

    返回 `[(face, point)]`，按输入序去重（`min_sep` 内只留第一个）。
    """
    out: list[tuple[str, tuple[float, float]]] = []
    for p in wire_pts or ():
        pt = (float(p[0]), float(p[1]))
        face, sd, _ax = face_of(pt, bbox)
        if sd < -inside_tol or sd > edge_tol:
            continue
        if any(math.hypot(pt[0] - q[0], pt[1] - q[1]) < min_sep for _f, q in out):
            continue
        out.append((face, pt))
    return out


# --------------------------------------------------------------------------- #
# ② 脚名指派（轴向规则 + 元件内 1:1 唯一）
# --------------------------------------------------------------------------- #
def assign_names(bbox, pts, texts, *, r: float = BOX_TEXT_R):
    """`pts` = `[(face, point)]`（`edge_points` 的输出）；返回 `[{point, face, content, dist}]`。

    规则 = 跳到"`点 -> 框内` 那条射线段"最近的一条文本（跳 0 的 `axis`，覆盖内 94.5%）；
    **1:1 唯一**：一条文本最多配一个点、一个点最多配一条文本，按距离从小到大贪心。
    ⚠ **不按 `cls` 过滤**（跳 0.5：数字脚名的 `pinName` 纯度只有 3.5%）。
    """
    pairs = _pairs(texts)
    if not pairs or not pts:
        return []
    w, h = max(bbox[2] - bbox[0], 1e-6), max(bbox[3] - bbox[1], 1e-6)
    L = 0.5 * min(w, h)
    cand = []
    for pi, (face, pt) in enumerate(pts):
        n = NORM_IN[face]
        seg_end = (pt[0] + n[0] * L, pt[1] + n[1] * L)
        for ti, (content, tb) in enumerate(pairs):
            d2, _t = seg_dist(_center(tb), pt, seg_end)
            if d2 <= r:
                cand.append((d2, pi, ti))
    cand.sort(key=lambda t: t[0])
    used_p, used_t, out = set(), set(), []
    for d2, pi, ti in cand:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        out.append({"point": pts[pi][1], "face": pts[pi][0],
                    "content": pairs[ti][0], "dist": d2})
    return out


# --------------------------------------------------------------------------- #
# ③ 号源：库 -> 投票表 -> 弃权
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def load_db(use_vote: bool = True):
    """`(lib, table)`：lib = 官方 IC 库（唯一名判据），table = GT 投票表（可关）。"""
    lib = pin_from_name.load_official()
    table = None
    if use_vote:
        from pcbmodel.integrate import ic_db
        if ic_db.DB_PATH.is_file():
            table = ic_db.load()
    return lib, table


def _norm_model(s: str) -> str:
    return "".join(str(s).split()).upper()


def lib_keys(lib) -> dict:
    """归一化后的库键 `{归一串: 原键}`（括号/空格差异在这一层吃掉）。"""
    return {_norm_model(k): k for k in (lib or {})}


def model_like(s: str, keys: dict) -> bool:
    """像不像型号：**在库里** + 够长 + 字母数字混合 + 不是位号形态。

    ⚠ 库里混着 `B`/`D`/`A`/`U1`/`OUT` 这类短键（来自官方 `searchName` 缺失的元件），
    不加这三道闸会把图上任意一个 `D` 文本当成型号 ⇒ 直接乱配脚号。
    """
    t = _norm_model(s)
    if len(t) < MODEL_MIN_LEN or t not in keys:
        return False
    if not (any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t)):
        return False
    return not _JUNK_KEY.match(t)


def _edit_le1(a: str, b: str) -> bool:
    """Levenshtein <=1: one insertion, deletion or substitution; no transposition."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def match_model(content: str, keys: dict, *, fuzzy1: bool = False):
    """Exact first; optionally accept one unique valid library key within one edit.

    Both the OCR token and the candidate retain all three shape gates.
    A missing or ambiguous candidate always abstains. This never rewrites OCR text.
    """
    norm = _norm_model(content)
    if model_like(norm, keys):
        return keys[norm]
    if not fuzzy1 or not model_like(norm, {norm: norm}):
        return None
    candidates = [k for k in keys if model_like(k, keys) and _edit_le1(norm, k)]
    return keys[candidates[0]] if len(candidates) == 1 else None


def resolve_models(bboxes, texts, *, keys=None, lib=None, r: float = MODEL_R,
                   fuzzy1: bool = False, reuse: bool = False):
    """给若干元件（按顺序）各配一个**型号串**；返回 `{idx: 型号串}`。

    `name` 在真实管线里**没人填**（`ocr-real` 路径的 box 全是 `name=None`，2026-09-23 实测），
    所以型号只能从图上的文本里认：**库里精确命中 + 型号形态**，再按"离框最近"排序做
    **全局 1:1** 贪心（一条文本最多给一个元件）⇒ 抢不到就**弃权**，不硬塞邻居的型号。
    fuzzy1=True 只增加库内唯一的一字符纠错候选；默认 False 保持精确匹配。

    `reuse=True`（**默认 False ⇒ 逐位不变**）：跑完上面那遍 1:1 后**再补一遍空位** ——
    第二遍**不再占用文本**（同一条文本可服务多个元件），但**每个元件仍只拿一个型号**。
    ⚠ **只补空**：第一遍已配上的元件**一律不改**（`used_b` 带着走）——
    实测这是关键的保守性，`ok_to_wrong` 恒 0。动机见 MODEL_HANDOFF 条目 95：
    同型号元件常只印一次型号，1:1 会让近者拿走、其余集体弃权。
    ⚠ **不许把第二遍写成"全量最近"**：那会让已有配对改挂到更近的（可能错的）文本上。
    """
    pairs = _pairs(texts)
    if not pairs or not bboxes:
        return {}
    if keys is None:
        lib = lib if lib is not None else load_db(True)[0]
        keys = lib_keys(lib)
    models = [match_model(content, keys, fuzzy1=fuzzy1) for content, _tb in pairs]
    scored = []
    for bi, bbox in enumerate(bboxes):
        for ti, (content, tb) in enumerate(pairs):
            if models[ti] is None:
                continue
            _f, sd, _ax = face_of(_center(tb), bbox)
            d = abs(sd)
            if d <= r:
                scored.append((d, bi, ti))
    scored.sort(key=lambda t: t[0])
    used_b, used_t, out = set(), set(), {}
    for d, bi, ti in scored:
        if bi in used_b or ti in used_t:
            continue
        used_b.add(bi)
        used_t.add(ti)
        out[bi] = models[ti]
    if not reuse:
        return out
    # 第二遍：**只补空**（文本可重复用；已配上的元件不进这一遍）
    for _d, bi, ti in scored:
        if bi in used_b:
            continue
        used_b.add(bi)
        out[bi] = models[ti]
    return out


def _bump(C, k, n: int = 1) -> None:
    """`Counter` 或普通 dict 都能计数（runner 传的是普通 dict）。"""
    C[k] = C.get(k, 0) + n


def build(bbox, name, wire_pts, texts, *, lib=None, table=None, use_vote: bool = True,
          edge_tol: float = BOX_EDGE_TOL, r: float = BOX_TEXT_R,
          min_agree: int = 2, n_pins: int | None = None,
          stats=None) -> list[DraftPin]:
    """三段全过才产脚；任一段不过就**少写**（不猜）。

    `use_vote=False` 关掉 GT 投票表（消融：只用官方库）。`stats` 供诊断计数。
    """
    C = stats if stats is not None else Counter()
    pts = edge_points(bbox, wire_pts, edge_tol=edge_tol)
    _bump(C, "pt_cand", len(pts))
    if not pts:
        return []
    named = assign_names(bbox, pts, texts, r=r)
    _bump(C, "name_cand", len(named))
    if not named:
        return []
    if lib is None:
        lib, table = load_db(use_vote)
    L = (lib or {}).get(str(name).strip()) if name else None
    if n_pins is None:
        n_pins = pin_from_name.package_n_pins(L)
    pins: list[DraftPin] = []
    for it in named:
        k, src = pin_from_name.number_for(name, it["content"], lib=lib, table=table,
                                          n_pins=n_pins, min_agree=min_agree)
        _bump(C, f"src_{src}")
        if k is None:
            continue
        pins.append(DraftPin(point=it["point"], pin_number_label=str(k),
                             pinname=it["content"]))
    _bump(C, "pins", len(pins))
    return pins


# --------------------------------------------------------------------------- #
# 自检（不依赖数据集/torch）
# --------------------------------------------------------------------------- #
def selftest() -> int:
    B = [0.0, 0.0, 100.0, 60.0]          # y-up 框：x∈[0,100], y∈[0,60]
    # ① 候选点：框内被拒、太远被拒、框外近处收下、近重复去重
    pts = edge_points(B, [(100.0 + 8.0, 30.0),         # R 面外 8px ✓
                          (100.0 + 8.5, 30.5),         # 与上一点 <3px ⇒ 去重
                          (50.0, 30.0),                # 框内 ⇒ 拒
                          (100.0 + 40.0, 30.0),        # 太远 ⇒ 拒
                          (50.0, -10.0)])              # B 面外 10px ✓
    assert len(pts) == 2, pts
    assert pts[0][0] == "R" and pts[1][0] == "B", pts
    assert edge_points(B, []) == [] and edge_points(B, None) == []

    # ② 指派：就近取、1:1 唯一、半径外不取
    texts = [("VCC", [104.0, 26.0, 130.0, 34.0]),      # 靠 R 面上那个点
             ("GND", [40.0, -18.0, 62.0, -6.0])]       # 靠 B 面下那个点
    got = assign_names(B, pts, texts, r=15.0)
    assert len(got) == 2, got
    assert {g["content"] for g in got} == {"VCC", "GND"}, got
    m = {g["content"]: g for g in got}
    assert m["VCC"]["face"] == "R" and m["GND"]["face"] == "B", got
    assert assign_names(B, pts, texts, r=1.0) == []    # 半径太小 ⇒ 全弃权
    #   一条文本只能配一个点（两个点抢同一个文本时只有一个拿到）
    one = assign_names(B, [(("R"), (100.0, 30.0)), (("R"), (100.0, 31.0))],
                       [("VCC", [102.0, 26.0, 120.0, 34.0])], r=15.0)
    assert len(one) == 1, one
    #   不按 cls 过滤：这里传的就是裸 (content, bbox)，没有类信息 —— 契约即"全类可用"

    # ③ 弃权：库里查不到 / 一对多且无票 / 没型号 ⇒ 一个脚都不写
    lib = {"UNAMB": {"VCC": ["1"]}, "POWER": {"VDD": ["7", "16"]}}
    pins = build(B, "UNAMB", [p for _f, p in pts], texts, lib=lib, table=None)
    assert len(pins) == 1 and pins[0].pin_number_label == "1", pins
    assert pins[0].pinname == "VCC"
    assert build(B, "POWER", [p for _f, p in pts], texts, lib=lib, table=None) == []
    assert build(B, None, [p for _f, p in pts], texts, lib=lib, table=None) == []
    assert build(B, "UNAMB", [], texts, lib=lib) == []            # 没导线点
    assert build(B, "UNAMB", [p for _f, p in pts], [], lib=lib) == []   # 没文本

    # ④ 硬约定：`PIN_OFFSET_PX` 里 `box` 只能**存在且为 None**（=`DERIVABLE_2PIN` 不含它）。
    #    给它一个数值 ⇒ `derive_pins` 每个 box 白送 2 个点（FP），见 HANDOFF 76。
    from pcbmodel.integrate import pin_geom
    assert pin_geom.PIN_OFFSET_PX.get("box", None) is None, "box 的偏移量必须是 None"
    assert "box" not in pin_geom.DERIVABLE_2PIN, "box 不许进 DERIVABLE_2PIN"
    assert pin_geom.derive_pins([0.0, 0.0, 100.0, 60.0], "box", 2) is None
    print("selftest OK（候选点/去重/指派唯一/半径/弃权三态/不碰 PIN_OFFSET_PX）")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
