# -*- coding: utf-8 -*-
"""阶段 2/3 孤立上限回环（`pin_geom --ceiling` 的方法论，搬到导线这一环）。

**为什么先做这个再训 U-Net**：输入是**官方 GT 导线掩码**（分割完美），所以量到的损失
**全部**来自「矢量化（2.2）+ 连通聚类（3.1）」这两步规则代码，与检测 / OCR / U-Net 无关。
先知道这两步的天花板，才知道 U-Net 的 IoU 该追到多少、后处理该往哪调。

三个数：

| 量 | 含义 | 用途 |
|---|---|---|
| **A 段还原率** | GT edges（97% 轴对齐，中位 21px）经「骨架→折线→2 点段」后剩几条、端点差多少 | 2.2 的验收口径；定 `--eps` / 最短段阈值 |
| **B netF1 落差** | 骨架段 vs **GT edges**（都用完美 GT 引脚点） | **向量化本身的代价** —— 若落差小，2.2 就不值得投入 |
| **C 容差曲线** | tol = 2/3/5/8/12px（5px = `cfg.line_tol`） | 断崖在哪边 ⇒ 推理期按「**连通性**」而非「IoU」扫阈值 |

**帧**：`load_sample` 给的 `wire_mask_up` 是 canonical **y-up**，`arr[y, x]` ↔ y-up 坐标
`(x, y)`，与官方标签同帧，**本模块不翻帧**（§八 风险 6/12）。

**已知不可恢复**：飞线连接点是**零长度段**（0.30%），骨架化后消失 —— 这是设计性损失，
不是 bug（MODEL_HANDOFF §五 2.2）。

跑（**pcbmodel env**，项目根 —— 需要 skimage + cv2，pcbeval env 里没有 skimage）:
  /home/linqs/.conda/envs/pcbmodel/bin/python -m pcbmodel.wire.ceiling --ceiling --limit 0
  /home/linqs/.conda/envs/pcbmodel/bin/python -m pcbmodel.wire.ceiling --selftest   # 合成图，不依赖数据集
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

PROJ = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ / "pcbmodel" / "runs" / "wire"
TOLS = (2.0, 3.0, 5.0, 8.0, 12.0)
GT_TOL = 5.0            # = `EvalConfig.line_tol`；GT-edges 基线与骨架行同 tol 才可比


# --------------------------------------------------------------------------- #
# 2.2 矢量化：掩码 → 骨架 → 折线 → 2 点段
# --------------------------------------------------------------------------- #
def skeleton_of(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(np.asarray(mask) > 0)


def _degree(sk: np.ndarray) -> dict[tuple[int, int], int]:
    """每个骨架像素的 8 邻域骨架像素数（端点=1、通路=2、交叉>=3）。"""
    pts = {(int(r), int(c)) for r, c in np.argwhere(sk)}
    deg = {}
    for r, c in pts:
        deg[(r, c)] = sum((r + dr, c + dc) in pts
                          for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    return deg


def skeleton_polylines(sk: np.ndarray) -> list[list[tuple[int, int]]]:
    """骨架 → 折线列表（每条形如 [(row, col), ...]）。

    规则：从「非通路像素」（deg != 2）出发，沿 deg == 2 的链走到下一个非通路像素。
    残余的**纯环**（全 deg == 2）单独补走一遍 —— 导线成环是真实的（电源环）。
    ponytail: 8 邻域下 2×2 方块会让 deg==2 的像素有两个互为邻接的邻居，取第一个即可；
    骨架化输出基本不产生这种形状，真出问题时再看。
    """
    deg = _degree(sk)
    nbrs = defaultdict(list)
    for (r, c) in deg:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in deg:
                    nbrs[(r, c)].append((r + dr, c + dc))
    used: set[frozenset] = set()

    def walk(start, second):
        path = [start, second]
        used.add(frozenset((start, second)))
        prev, cur = start, second
        while deg[cur] == 2:
            nxt = [q for q in nbrs[cur] if q != prev]
            if not nxt or frozenset((cur, nxt[0])) in used:
                break
            used.add(frozenset((cur, nxt[0])))
            path.append(nxt[0])
            prev, cur = cur, nxt[0]
        return path

    polys: list[list[tuple[int, int]]] = []
    for s in [p for p in deg if deg[p] != 2]:
        for n in nbrs[s]:
            if frozenset((s, n)) not in used:
                polys.append(walk(s, n))
    for s in deg:                                   # 纯环残余
        for n in nbrs[s]:
            if frozenset((s, n)) not in used:
                polys.append(walk(s, n))
    return [p for p in polys if len(p) >= 2]


def polyline_to_segments(path: list[tuple[int, int]], eps: float = 1.0,
                         min_len: float = 2.0) -> list[tuple]:
    """折线 → 2 点段（像素 (row,col) → y-up 坐标 (x, y)）。

    先 RDP 简化成折点序列（97% 的 GT edge 是轴对齐直线，RDP 正好还原成直段），
    再相邻两点成段；**短于 `min_len` 的段丢弃**（骨架化在端点和交叉处会长小毛刺）。
    """
    import cv2
    pts = np.array([[c, r] for r, c in path], dtype=np.int32).reshape(-1, 1, 2)
    simp = cv2.approxPolyDP(pts, float(eps), False).reshape(-1, 2).astype(float)
    out = []
    for i in range(len(simp) - 1):
        a, b = simp[i], simp[i + 1]
        if float(np.hypot(*(b - a))) >= min_len:
            out.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return out


def mask_to_segments(mask: np.ndarray, eps: float = 1.0,
                     min_len: float = 2.0) -> list[tuple]:
    segs: list[tuple] = []
    for path in skeleton_polylines(skeleton_of(mask)):
        segs.extend(polyline_to_segments(path, eps=eps, min_len=min_len))
    return segs


# --------------------------------------------------------------------------- #
# 3.1 连通聚类（几何并查集）→ netF1
# --------------------------------------------------------------------------- #
def _d2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _pt_seg_d2(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    if L == 0:
        return _d2(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return (p[0] - a[0] - t * dx) ** 2 + (p[1] - a[1] - t * dy) ** 2


def cluster_components(segs: list[tuple], pins: dict,
                       tol: float) -> list[tuple[set, list[int]]]:
    """几何连通分量：**端点-端点** 与 **端点-线段（T 型）** 两种接触都算连通。

    `pins` = {(desig, pin_key): (x, y)}；引脚点落在段上（含端点）即并入该分量。
    返回 `[(该分量的 pin 键集合, 该分量的段下标列表), ...]` —— 段下标是给 `lineF1`
    用的（lineF1 只在**已配对的 net 内**比 2 点段，见 `scorer.line_score`）。

    ponytail: 段两两 O(n²)。单例中位 124 段、p99 ~660 → 最坏 4×10⁵ 次距离，够快。
    若某天单例上千段，再上网格分桶（按 tol 分箱只比邻近箱）。
    """
    keys = list(pins)
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    t2 = tol * tol
    for i in range(len(segs)):
        a, b = segs[i]
        for j in range(i + 1, len(segs)):
            c, d = segs[j]
            if (min(_d2(a, c), _d2(a, d), _d2(b, c), _d2(b, d)) <= t2
                    or _pt_seg_d2(c, a, b) <= t2 or _pt_seg_d2(d, a, b) <= t2
                    or _pt_seg_d2(a, c, d) <= t2 or _pt_seg_d2(b, c, d) <= t2):
                uni(("s", i), ("s", j))
    for k in keys:
        for i, (a, b) in enumerate(segs):
            if _pt_seg_d2(pins[k], a, b) <= t2:
                uni(("p", k), ("s", i))

    comp: dict = defaultdict(lambda: (set(), []))
    for k in keys:
        comp[find(("p", k))][0].add(k)
    for i in range(len(segs)):
        comp[find(("s", i))][1].append(i)
    # ⚠ **单 pin 网要出**（官方单 pin 网占 34.3%，漏了这部分等于白扔 netF1）
    # 无引脚的孤立导线分量不产出（其 hyperGraph 为空，配不上任何 GT net）
    return [(p, s) for p, s in comp.values() if p]


def build_pred_nets(clusters: list[tuple], tok_of: dict, segs: list[tuple]) -> dict:
    """连通分量 → pred `nets`。

    - token 文本**取自 GT hyperGraph 原样**（保证逐字相等，不自己拼 `desig.pin`）；
    - `edges` 用官方 2 点格式（`[{"x","y"},{"x","y"}]`）⇒ 可交给 `scorer.line_score`。
    """
    out = {}
    for n, (keys, idxs) in enumerate(sorted(clusters, key=lambda c: -len(c[0]))):
        toks = [tok_of[k] for k in sorted(keys) if k in tok_of]
        if not toks:
            continue
        edges = {f"E_{m}": [{"x": a[0], "y": a[1]}, {"x": b[0], "y": b[1]}]
                 for m, (a, b) in enumerate(segs[i] for i in idxs)}
        out[f"net_{n}"] = {"hyperGraph": "(" + ",".join(toks) + ")", "edges": edges}
    return out


def gt_pins_of(target: dict) -> dict:
    pins = {}
    for desig, pd in target.get("pins", {}).items():
        for key, v in pd.items():
            pt = v.get("point") or {}
            if "x" in pt and "y" in pt:
                pins[(desig, key)] = (float(pt["x"]), float(pt["y"]))
    return pins


def gt_segments_of(target: dict) -> list[tuple]:
    out = []
    for net in target.get("nets", {}).values():
        for seg in net.get("edges", {}).values():
            if len(seg) != 2:
                continue
            a = (float(seg[0]["x"]), float(seg[0]["y"]))
            b = (float(seg[1]["x"]), float(seg[1]["y"]))
            out.append((a, b))
    return out


def token_map(target: dict) -> tuple[dict, int]:
    """{(desig, pin_key): GT hyperGraph 原 token 文本} + 解析不出的 token 数。

    ⚠ 用 `pcbeval.refs.TokenResolver`（对 components 键集最长前缀），**不按点拆**
    （§八 风险 10：`U7.0618.1` / `U2.3.438` 这类反例）。
    """
    from pcbeval.refs import TokenResolver, iter_tokens
    R = TokenResolver(list(target.get("components", {})), target.get("pins", {}))
    m, bad = {}, 0
    for net in target.get("nets", {}).values():
        for tk in iter_tokens(net.get("hyperGraph", "")):
            got = R.resolve(tk)
            if got is None:
                bad += 1
            else:
                m[got] = tk
    return m, bad


# --------------------------------------------------------------------------- #
# 测量
# --------------------------------------------------------------------------- #
def _endpoint_nn(a: list[tuple], b: list[tuple]) -> tuple[float, float]:
    """a 的端点到 b 的端点的最近距离（中位 / p90）。"""
    if not a or not b:
        return float("nan"), float("nan")
    from scipy.spatial import cKDTree
    ea = np.array([p for s in a for p in s])
    eb = np.array([p for s in b for p in s])
    d, _ = cKDTree(eb).query(ea)
    return float(np.median(d)), float(np.percentile(d, 90))


def measure(limit: int = 0, tols=TOLS, eps: float = 1.0, min_len: float = 2.0,
            ids: list[str] | None = None, mask_of=None) -> dict:
    """`mask_of(sample) -> y-up 0/1 uint8` 给了就用**预测掩码**（阶段 2.1 验收），否则用 GT 掩码。

    预测档必须 `with_mask=False` 读样本 —— 从源头保证 **GT 掩码没被偷用**
    （GT 基线行不受影响：`gt_segments_of` 读的是 target JSON，与掩码无关）。
    `ids` 用来把评测限定在**干净 val** 上，与 GT 掩码的上限同集对比。
    """
    from pcbeval.config import EvalConfig
    from pcbeval.scorer import line_score, net_hypergraph_score
    from pcbmodel.data.loader import load_sample
    from pcbmodel.integrate.runner import case_ids

    cfg = EvalConfig()
    ids = list(ids) if ids is not None else (
        case_ids("all")[:limit] if limit else case_ids("all"))
    acc = {("skeleton", t): [0, 0, 0] for t in tols}      # tp, fp, fn
    acc[("gt-edges", GT_TOL)] = [0, 0, 0]
    line_acc = {"skeleton": [0, 0, 0], "gt-edges": [0, 0, 0]}
    n_gt_seg = n_ext_seg = n_bad_tok = 0
    fwd, bwd = [], []
    per_case = []
    for cid in ids:
        s = load_sample(cid, source="official-train", with_mask=mask_of is None, cfg=cfg)
        t = s.target
        tok_of, bad = token_map(t)
        n_bad_tok += bad
        pins = gt_pins_of(t)
        gt_segs = [sg for sg in gt_segments_of(t) if sg[0] != sg[1]]   # 零长度飞线单列
        m_up = mask_of(s) if mask_of is not None else s.wire_mask_up
        ext = mask_to_segments(m_up, eps=eps, min_len=min_len)
        n_gt_seg += len(gt_segs)
        n_ext_seg += len(ext)
        m1, p1 = _endpoint_nn(ext, gt_segs)
        m2, p2 = _endpoint_nn(gt_segs, ext)
        fwd.append(m1); bwd.append(m2)
        per_case.append({"case": cid, "gt_seg": len(gt_segs), "ext_seg": len(ext),
                         "nn_ext→gt_med": round(m1, 2), "nn_gt→ext_med": round(m2, 2),
                         "p90_gt→ext": round(p2, 2)})
        for tol in tols:
            pred = {"nets": build_pred_nets(
                cluster_components(ext, pins, tol), tok_of, ext)}
            r = net_hypergraph_score(pred, t)
            acc[("skeleton", tol)][0] += r["tp"]
            acc[("skeleton", tol)][1] += r["fp"]
            acc[("skeleton", tol)][2] += r["fn"]
            if tol == GT_TOL:            # lineF1 只在交付 tol 上比（口径 = cfg.line_tol）
                lr = line_score(pred, t, r["matched"], cfg)
                for i, k in enumerate(("tp", "fp", "fn")):
                    line_acc["skeleton"][i] += lr[k]
        pred = {"nets": build_pred_nets(
            cluster_components(gt_segs, pins, GT_TOL), tok_of, gt_segs)}
        r = net_hypergraph_score(pred, t)
        acc[("gt-edges", GT_TOL)][0] += r["tp"]
        acc[("gt-edges", GT_TOL)][1] += r["fp"]
        acc[("gt-edges", GT_TOL)][2] += r["fn"]
        lr = line_score(pred, t, r["matched"], cfg)
        for i, k in enumerate(("tp", "fp", "fn")):
            line_acc["gt-edges"][i] += lr[k]

    def f1(v):
        tp, fp, fn = v
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    rows = [{"tol_px": t, "net_f1": round(f1(acc[("skeleton", t)]), 4)} for t in tols]
    base = {"tol_px": GT_TOL, "net_f1": round(f1(acc[("gt-edges", GT_TOL)]), 4)}
    return {"n_cases": len(ids), "eps": eps, "min_len": min_len,
            "mask_source": "gt" if mask_of is None else "pred",
            "gt_segments": n_gt_seg, "extracted_segments": n_ext_seg,
            "seg_ratio": round(n_ext_seg / n_gt_seg, 3) if n_gt_seg else 0.0,
            "unresolved_tokens": n_bad_tok,
            "line_f1": {k: round(f1(v), 4) for k, v in line_acc.items()},
            "nn_ext_to_gt_median_px": round(float(np.nanmedian(fwd)), 3) if fwd else None,
            "nn_gt_to_ext_median_px": round(float(np.nanmedian(bwd)), 3) if bwd else None,
            "rows": rows, "baseline_gt_edges_tol5": base, "worst_cases": sorted(
                per_case, key=lambda c: c["p90_gt→ext"], reverse=True)[:5]}


def render_md(res: dict) -> str:
    b = res["baseline_gt_edges_tol5"]["net_f1"]
    pred = res.get("mask_source", "gt") == "pred"
    row_name = "**U-Net 预测掩码**" if pred else "骨架抽出段"
    L = ["# 阶段 2.1 真管线（输入 = **U-Net 预测掩码**）" if pred else
         "# 阶段 2/3 孤立上限回环（输入 = **官方 GT 导线掩码**）", "",
         f"> {res['n_cases']} 例｜矢量化参数 `eps={res['eps']}px`、"
         f"最短段 `{res['min_len']}px`", "",
         "## A 段还原率（2.2 的验收口径）", "",
         "| 量 | 值 |", "|---|---:|",
         f"| GT edges（去零长度飞线） | {res['gt_segments']} |",
         f"| 骨架→折线→2 点段 抽出 | {res['extracted_segments']} |",
         f"| **段数比（抽/GT）** | **{res['seg_ratio']:.3f}** |",
         f"| 抽出段端点 → 最近 GT 端点（中位） | {res['nn_ext_to_gt_median_px']} px |",
         f"| GT 端点 → 最近抽出段端点（中位） | {res['nn_gt_to_ext_median_px']} px |",
         f"| hyperGraph token 解析不出的数 | {res['unresolved_tokens']} |", "",
         "## B/C netF1 + lineF1（都用**完美 GT 引脚点**）", "",
         "| 段来源 | tol | netF1 | 0.35×netF1（分） | lineF1 | 0.10×lineF1（分） |",
         "|---|---:|---:|---:|---:|---:|",
         f"| **GT edges（理想矢量化）** | {GT_TOL:.0f}px | **{b:.4f}** | {35 * b:.1f} | "
         f"**{res['line_f1']['gt-edges']:.4f}** | {10 * res['line_f1']['gt-edges']:.1f} |"]
    sk = res["line_f1"]["skeleton"]
    for r in res["rows"]:
        mark = " ◀ 交付 tol" if abs(r["tol_px"] - GT_TOL) < 1e-9 else ""
        L.append(f"| {row_name} | {r['tol_px']:.0f}px{mark} | {r['net_f1']:.4f} | "
                 f"{35 * r['net_f1']:.1f} | "
                 + (f"{sk:.4f} | {10 * sk:.1f} |" if mark else "— | — |"))
    L += ["", "## 读法", ""]
    if pred:
        L.append("- ⚠ **这一行同时含分割误差与矢量化误差**（GT 掩码行才是纯矢量化的代价）"
                 "⇒ 读差值时必须跟同集数的 GT 掩码报告对照，别单独解读。")
    else:
        L.append(f"- **向量化的代价**（同 tol {GT_TOL:.0f}px）：netF1 "
                 f"{res['rows'][[r['tol_px'] for r in res['rows']].index(GT_TOL)]['net_f1']:.4f} vs "
                 f"{b:.4f}；lineF1 {sk:.4f} vs {res['line_f1']['gt-edges']:.4f}。"
                 "落差小 ⇒ 2.2 的规则代码够用，U-Net 只管把掩码画对。")
    L += [f"- **段数比 {res['seg_ratio']:.3f}**：GT 把长导线拆成一串短 2 点段（中位 21px），"
          "而 RDP 会把共线的串合成一条 → **段数天然偏少**，这部分损失落在 lineF1 上"
          "（官方 edges 数量严格比对），**不影响 netF1**（netF1 只看 token 集合）。",
          "- **断崖**决定后处理方向：曲线在某 tol 之后掉 ⇒ 该按**连通性**扫阈值/膨胀，"
          "而不是按 IoU 调分割阈值。",
          ("- 掩码来自 U-Net 预测，但**引脚点与 token 文本仍取自 GT** ⇒ 这不是端到端分，"
           "只隔离了阶段 2.1 的贡献。" if pred else
           "- 这是**孤立上限**（完美分割 + 完美引脚 + token 文本取自 GT）—— 真实管线还要乘上"
           "检测框误差与 token 覆盖率（OCR），见 MODEL_HANDOFF §九。"), ""]
    if res["worst_cases"]:
        L += ["## 最差的 5 例（p90 端点差）", "",
              "| case | GT 段 | 抽出段 | 抽→GT 中位 | GT→抽 中位 | p90 |",
              "|---|---:|---:|---:|---:|---:|"]
        for c in res["worst_cases"]:
            L.append(f"| `{c['case']}` | {c['gt_seg']} | {c['ext_seg']} | "
                     f"{c['nn_ext→gt_med']} | {c['nn_gt→ext_med']} | {c['p90_gt→ext']} |")
        L.append("")
    return "\n".join(L)


def selftest() -> int:
    """合成图自检：L 形折线 + 一条交叉线 + 一个孤立引脚。不依赖数据集。"""
    m = np.zeros((200, 200), np.uint8)
    m[50, 20:150] = 1                      # 横线 (20,50)-(149,50)
    m[50:120, 100] = 1                     # 竖线 (100,50)-(100,119) → T 型交叉
    m[20:40, 180] = 1                      # 孤立竖线 (180,20)-(180,39)
    segs = mask_to_segments(m)
    assert segs, "骨架应抽出线段"
    # T 型交叉：横线被切成 2 段、竖线 1 段、孤立 1 段 = 4 段
    assert len(segs) == 4, f"应抽 4 段（T 交叉把横线切成两段），实得 {len(segs)}: {segs}"
    ends = {tuple(round(v) for v in p) for s in segs for p in s}
    for want in [(20, 50), (149, 50), (100, 119), (180, 20), (180, 39)]:
        assert want in ends, f"端点 {want} 丢失；实得 {sorted(ends)}"
    # 连通性：T 型交叉处 tol=5 应把横竖两段并成一条 net（引脚能追到该分量）
    clusters = cluster_components(segs, {("U1", "pin_1"): (105.0, 50.0)}, 5.0)
    assert any(("U1", "pin_1") in p for p, _ in clusters), "引脚应并入所在分量"
    # 引脚在孤立竖线上（x=180）→ 与横线不相连，只跟孤立线同分量
    alone = cluster_components(segs, {("U2", "pin_1"): (180.0, 25.0)}, 2.0)
    got = [p for p, _ in alone if ("U2", "pin_1") in p]
    assert len(got) == 1, "孤立引脚应恰好属于一个分量"
    # lineF1 要的段下标：无引脚的分量**被丢弃**（其 hyperGraph 为空、配不上 GT net），
    # 但留下的下标必须无重、且带引脚的分量必须带段。
    flat = [i for _, s in clusters for i in s]
    assert len(flat) == len(set(flat)), "段下标不得重复"
    assert all(s for _, s in clusters), "带引脚的分量必须带段"
    pred = build_pred_nets(clusters, {("U1", "pin_1"): "U1.50"}, segs)
    assert len(next(iter(pred.values()))["edges"]) >= 2, "net 应带上自己的 2 点段"
    print("[wire.ceiling] selftest PASS：4 段 / 端点齐 / T 型交叉合并 / 引脚归属")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="wire.ceiling",
                                 description="阶段 2/3 孤立上限回环（GT 掩码 → 骨架 → netF1）")
    ap.add_argument("--ceiling", action="store_true", help="跑上限回环并出报告")
    ap.add_argument("--selftest", action="store_true", help="合成图自检（不需数据集）")
    ap.add_argument("--limit", type=int, default=40, help="跑多少例（0=全部 200）")
    ap.add_argument("--eps", type=float, default=1.0,
                    help="RDP 简化容差 px。实测（60 例）：eps 1/2/4 的 netF1 全等 0.704，"
                         "lineF1 0.667/0.637/0.633、段数比 0.795/0.759/0.753 → **取 1.0**"
                         "（像素栅格上 45° 台阶的垂直偏差 ≈0.71px，eps<1 会开始锯齿）")
    ap.add_argument("--min-len", type=float, default=2.0, help="最短段 px（去骨架毛刺）")
    ap.add_argument("--ckpt", default=None,
                    help="U-Net 权重（`seg/train_seg.py` 的 best.pt）：给了就用**预测掩码**替代 "
                         "GT 掩码，即阶段 2.1 的验收跑法。报告落到 `runs/wire/pred.md`")
    ap.add_argument("--ids", default=None,
                    help="逗号分隔 case 列表（默认全部）。预测档应传干净 val，才能与 GT 掩码上限同集比")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thr", type=float, default=0.5, help="掩码二值化阈值（预测档）")
    ap.add_argument("--floor", action="store_true",
                    help="**底分探针**：全零掩码（一根线不画）⇒ 每脚自成一网。"
                         "§八 风险 16 那个「~17 分免费底分」此前只有口述、没有产物，用它归档")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.ceiling:
        ap.print_help()
        return 0

    ids = [x for x in (a.ids or "").split(",") if x] or None
    mask_of = None
    if a.floor:
        # 全零掩码 ⇒ `mask_to_segments` 出 0 段 ⇒ `build_pred_nets` 把**每个引脚自成一网**。
        # 这正是「不做任何事能拿到的 net 分」（§八 风险 16），用现成 measure 即可，无需改它。
        import numpy as np
        # ⚠ 用 `image_up` 的形状，**不是** `wire_mask_up` —— 预测档读样本时 `with_mask=False`
        #（`measure` 的规矩：防偷 GT 掩码），那里 `wire_mask_up` 是 None，`zeros_like(None)` 会出 0 维数组。
        mask_of = lambda s: np.zeros(s.image_up.shape[:2], dtype=np.uint8)   # noqa: E731
    elif a.ckpt:
        from pcbmodel.seg.unet import load_ckpt, predict_mask
        net = load_ckpt(a.ckpt, device=a.device)
        print(f"[wire.ceiling] 预测档：ckpt={a.ckpt} device={a.device} thr={a.thr}")

        def mask_of(s):                       # noqa: E306  闭包捕获 net
            return predict_mask(net, s.image_up, device=a.device, thr=a.thr)

    res = measure(limit=a.limit, eps=a.eps, min_len=a.min_len, ids=ids, mask_of=mask_of)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ⚠ 子集跑**不许覆盖权威报告**：`--limit 5` 的冒烟曾把 200 例的 `ceiling.md` 冲掉。
    # 只有"全集"这一种跑法写 `ceiling.md`/`pred.md`，其余带例数后缀。
    stem = "floor" if a.floor else ("pred" if mask_of else "ceiling")
    if a.limit or ids:
        stem = f"{stem}_{res['n_cases']}"
    (OUT_DIR / f"{stem}.md").write_text(render_md(res), encoding="utf-8")
    import json
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.floor:
        print(f"[wire.ceiling] ⚠ 底分探针（全零掩码）：{res['n_cases']} 例 ｜ "
              f"netF1={res['rows'][0]['net_f1']:.4f} = **{35 * res['rows'][0]['net_f1']:.1f} 分**"
              f"（§八 风险 16；引用 netF1 前先减它）")
    print(f"[wire.ceiling] {res['n_cases']} 例〔{res['mask_source']}〕｜ 段数比 "
          f"{res['seg_ratio']:.3f} ｜ netF1(GT edges)={res['baseline_gt_edges_tol5']['net_f1']:.4f}")
    for r in res["rows"]:
        print(f"   骨架 tol={r['tol_px']:>4.0f}px  netF1={r['net_f1']:.4f}  "
              f"→ {35 * r['net_f1']:.1f} 分")
    print(f"   -> {OUT_DIR / f'{stem}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
