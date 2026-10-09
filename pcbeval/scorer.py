# -*- coding: utf-8 -*-
"""pred vs GT 四类 F1 + 加权分（本地官方公式复现，P0 步骤 4）。

SingleCase = 100 × ( 0.3·ComponentF1 + 0.25·PinF1 + 0.35·NetHypergraphF1 + 0.1·NetLineF1 )
bbox 容差 20px；pin/line 端点 5px；字符串严格（10k ≠ 10K）；net 用 pin 集合 1:1 匈牙利。

匹配规则（best-effort，官方代码到手后用隔离小函数替换 —— config 注释/校准清单）：
  Component : 按 designator key；双方都有 key 的元件需 type 相等 + bbox 四角≤tol +
              文本(按 type: ic/connector…比 name，R/C/L/led/diode…比 value；GT 该字段为
              null 时跳过)。通过=TP；仅 pred 有=FP、仅 GT 有=FN；key 双方有但条件不满足
              计 FP+FN 各一（对称惩罚）。失败桶 key_miss/extra/wrong_type/wrong_bbox/
              wrong_text 供诊断。
  Pin       : 在 component key 双方都有的元件内，对 GT pin↔pred pin 按 point≤tol 做 1:1
              最大基数匹配；配对上还要求 GT/Pred 的 pin key（即 pin number，可含字母如
              pin_A5）严格一致 + pinname 严格相等（GT pinname null →
              `pinname_skip_when_gt_null` 跳过）。2026-09-09 修复：旧实现只查 pinname，
              编号写错但坐标对会被算成 TP。
  Net hyp   : 每 net 的 hyperGraph → pin-token 集合（字符串，net key 忽略）；pred/gt net
              间按 token 交集做 1:1 匈牙利（cost = -|交集|），交集>0 才算配对。
              TP=配对交集 token 数；配对 net 内 pred 独有 token=FP、gt 独有 token=FN，
              未配对 net 整体计 FP/FN。2026-09-09 修复：旧实现漏计配对 net 内差值，
              GT=(A,B) vs pred=(A) 也会被算成 F1=1.0。
  Line      : 只在 Net hyp 配对成功(交集>0)的 net 对内比线段；每对内 2 点线段按
              “两端点≤tol（允许反向）”1:1 最大基数匹配；未配对 net 的线段不计入（口径待官方校准）。

**宽松口径**（`EvalConfig.loose`，默认空 = 上面这套 strict 行为**逐字不变**）：

  L1 `"bjt"`  `bjt` ≡ `bjt_npn` ≡ `bjt_pnp`（官方答疑原话，决策 16）—— `_type_ok()`
  L2 `"gnd"`  `gnd` 的**位号序号不比对**（官方答疑原话，决策 17）—— `align_loose()`

L2 的实现是**把 pred 的松散 key 改挂到 GT 的 key 上**（同 type 内按 bbox 中心匈牙利配对），
下游严格 key 的代码原样复用。⚠ **只改 `components`/`pins` 的 key，不改 hyperGraph token** ——
「序号」包不包括 **pin 号**正是决策 17 待确认 ①（L3，差 3 分），所以这里保守：与 §四 实测
表格一致（L2 只动 compF1、pin/net/line 不动）。L3–L5 官方未答，**未实装**。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import (EvalConfig, TEXT_NAME_TYPES, TEXT_VALUE_TYPES, comp_text)

# 元件文本比较字段（best-effort，按官方“IC 比 Name、R·C 比 value”展开）。
# 词表统一在 config（含官方 41–43 类小写词表），此处仅别名，避免两处漂移。
_TEXT_NAME = TEXT_NAME_TYPES
_TEXT_VALUE = TEXT_VALUE_TYPES

# 宽松口径的**特性注册表**。⚠ 只收官方**已确认**的（决策 16/17）；L3–L5 官方未答
# （§五 D1），等答疑回来往下面两张表里各加一行即可，**不动其他代码**。
_LOOSE_BJT_FAMILY = frozenset({"bjt", "bjt_npn", "bjt_pnp"})   # L1：族内互认
_LOOSE_SERIAL_TYPES = frozenset({"gnd"})                        # L2：位号序号不比对
# ⚠⚠ **未确认**特性：GT 的元件键带 `∅` 时，key 不参与配对，改按 type+bbox 匈牙利配。
# 与 L2 是**同一条机制**（`align_loose`），差别只在覆盖面：
#   * `gnd` = 官方**已确认**（L2），进 `LOOSE_CONFIRMED`；
#   * 本题 = **09-19 问题单问题 1/2 里官方还没答的**，所以**单列、不进 CONFIRMED**。
# 实测 ∅ 元件 4034/9171 = 44.0%；**100% 是 ∅ 的 type 有 15 个**（gnd 2136 / net_input 323
# / net_short 163 / net_output 72 / other 66 / m3螺丝 58 / buzzer 21 / battary 21 /
# block 13 / net_bidirection 13 / jumper 9 / motor 8 / esd 6 / dc 6 / transfomer 5），
# `v` 99.9%（1092/1093）⇒ 覆盖 **99.5%** 的 ∅ 元件。
# 判据取「键以 ∅ 开头」而**不是**那串类型名 —— 名单会随数据漂，键前缀不会。
_LOOSE_EMPTY = "empty"
_EMPTY_MARK = "∅"
# 报告里要传的那一组（**只有 L1+L2**）。**调用点别手写字符串**，从这里取。
LOOSE_CONFIRMED: frozenset[str] = frozenset({"bjt"} | set(_LOOSE_SERIAL_TYPES))


# ---- 小工具 --------------------------------------------------------------
def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2.0 * tp / denom


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pt(p) -> tuple[float, float]:
    return float(p["x"]), float(p["y"])


def _seg_pt(seg, idx) -> tuple[float, float]:
    return _pt(seg[idx])


def _seg_feasible(ps, gs, tol: float) -> bool:
    """两端点都≤tol，允许 pred/gt 方向一致或反向。"""
    fwd = _d(_seg_pt(ps, 0), _seg_pt(gs, 0)) <= tol and _d(_seg_pt(ps, 1), _seg_pt(gs, 1)) <= tol
    rev = _d(_seg_pt(ps, 0), _seg_pt(gs, 1)) <= tol and _d(_seg_pt(ps, 1), _seg_pt(gs, 0)) <= tol
    return fwd or rev


def _bbox_ok(pbox, gbox, tol: float) -> bool:
    """4 角对应距离全部 ≤tol（y 翻转等距，pred/gt 同框即可）。"""
    return (_d((pbox[0], pbox[1]), (gbox[0], gbox[1])) <= tol and
            _d((pbox[2], pbox[1]), (gbox[2], gbox[1])) <= tol and
            _d((pbox[0], pbox[3]), (gbox[0], gbox[3])) <= tol and
            _d((pbox[2], pbox[3]), (gbox[2], gbox[3])) <= tol)


def _max_cardinality(costs) -> tuple[list, int]:
    """cost 矩阵 0=可行 1=不可行，返回 (匹配对列表, 可行配对数)。"""
    costs = np.asarray(costs, dtype=float)
    r, c = linear_sum_assignment(costs)
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if costs[i, j] < 0.5]
    return pairs, len(pairs)


def _hyper_tokens(hyper: str) -> list[str]:
    """hyperGraph `(tok,tok,...)` -> token 列表。

    官方语义 (comp.pin,...)，designator 可含点（如 U2.3 + pin 2 → token `U2.3.2`），
    不能靠正则猜拆（旧 `_TOK_RE` 会把 `U2.3.2` 截成 `U2.3`、丢 pin 号）。
    与 sanity._hyper_tokens 同口径：按逗号切分、不拆 token。
    """
    s = hyper.strip()
    if not s:
        return []
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return [tk.strip() for tk in s.split(",") if tk.strip()]


def _type_ok(ptype, gtype, cfg: EvalConfig) -> bool:
    """type 判据。L1（`"bjt"` in loose）：官方答疑明文 `bjt` ≡ `bjt_npn` ≡ `bjt_pnp`。"""
    if ptype == gtype:
        return True
    return ("bjt" in cfg.loose
            and ptype in _LOOSE_BJT_FAMILY and gtype in _LOOSE_BJT_FAMILY)


def _ctr(bbox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _retoken_nets(nets: dict, pc: dict, gc: dict, pred_pins: dict, gt_pins: dict,
                  ren: dict, cfg: EvalConfig) -> dict:
    """arm3：把 pred 的 hyperGraph token 也重挂到 GT 的键上（`cfg.token_align=True`）。

    - **comp 键**：用 `align_loose` 已经算好的 `ren`（同 type + bbox 中心匈牙利）。
    - **pin 键**：只能按**位置**配 —— ∅ 元件的 pin 号也是源工具计数器，没有可比字符串。
      每个 (pred 元件 → GT 元件) 对内按 `point_tol` 做 1:1 最大基数匹配（同 `pin_score` 的判据），
      **配不上的 token 原样保留**（老实记 FP/FN，不硬凑）。
    - token 解析一律走 `refs.TokenResolver`（最长前缀），**不按点拆**（§八 风险 10）。
    - 只有真的改了键才重建字符串，未改动的 token 逐字节保持原样（避免格式漂移）。
    """
    from pcbeval.refs import TokenResolver, iter_tokens, join_token, make_hypergraph
    resolver = TokenResolver(list(pc), {k: (pred_pins.get(k) or {}) for k in pc})
    pin_ren: dict[tuple[str, str], str] = {}
    for a, b in ren.items():
        pp = list((pred_pins.get(a) or {}).items())
        gg = list((gt_pins.get(b) or {}).items())
        if not pp or not gg:
            continue
        costs = [[0.0 if _d(_pt(p["point"]), _pt(g["point"])) <= cfg.point_tol else 1.0
                  for _gk, g in gg] for _pk, p in pp]
        pairs, _n = _max_cardinality(costs)
        for gi, pi in pairs:
            pin_ren[(a, pp[pi][0])] = gg[gi][0]
    out: dict = {}
    for nm, net in (nets or {}).items():
        toks = []
        for tk in iter_tokens(net.get("hyperGraph", "")):
            got = resolver.resolve(tk)
            if got is None:
                toks.append(tk)
                continue
            ck, pk = got
            ck2, pk2 = ren.get(ck, ck), pin_ren.get((ck, pk), pk)
            # ⚠ 用 `join_token` 回写：官方 token 是**去 `pin_` 前缀**形态（`∅216.213`），
            #   而 `resolve()` 返回的 pin 键**带**前缀 ⇒ 直接 f"{ck}.{pk}" 会拼出 `pin_pin_1`
            #   （2026-09-22 自检抓到）。未改动时保留原串，保证往返逐字节恒等。
            toks.append(join_token(ck2, pk2) if (ck2 != ck or pk2 != pk) else tk)
        out[nm] = {**net, "hyperGraph": make_hypergraph(toks)}
    return out


def align_loose(pred: dict, gt: dict, cfg: EvalConfig) -> dict:
    """L2（`"gnd"` in loose）：把 pred 的松散 key **改挂到 GT 的 key 上**。

    官方原话「`gnd` 的序号不影响评分」⇒ 序号不参与配对，改为**同 type 内按 bbox 中心
    的匈牙利配对**（1:1、最小化总距离）。只对「pred 有 GT 没」「GT 有 pred 没」的 key
    做重挂 —— **已经严格对上的 key 一律不动**。

    ⚠ **只改 key，不放松任何判据**：配错了下游 `_bbox_ok`/`_text_ok` 照样判 FP+FN，
    所以这**不会凭空造 TP**（bbox 本身就是配对的天然上限）。
    ⚠ **默认不改 `nets`**（hyperGraph token 里的序号原样保留）—— 见模块 docstring 的 L3 说明。
    **`cfg.token_align=True` 时才一起重挂**（arm3，见 `_retoken_nets`）。
    ⚠ 副产品：返回的 pred 里 `components` 的 key 与 token 里的位号**会不一致**，这是**故意的**，
    只为喂 scorer；**别把它当产物写盘**。
    """
    types = _LOOSE_SERIAL_TYPES & set(cfg.loose)
    empty = _LOOSE_EMPTY in set(cfg.loose)          # ∅ 键也按 type+bbox 配（**未确认**）
    if not types and not empty:
        return pred
    pc, gc = pred.get("components") or {}, gt.get("components") or {}

    def g_ok(k, t) -> bool:
        """该 GT 元件的 key 是否"不参与配对"。"""
        return t in types or (empty and k.startswith(_EMPTY_MARK))

    def p_ok(t) -> bool:
        """该 type 的 pred 元件是否可参与重挂。没有 GT 侧同伴时下面的循环自然跳过。"""
        return t in types or empty

    ren: dict[str, str] = {}
    for t in sorted({c.get("type", "") for c in gc.values()}):
        if not p_ok(t):
            continue
        pfree = sorted(k for k, c in pc.items() if c.get("type") == t and k not in gc)
        gfree = sorted(k for k, c in gc.items()
                       if c.get("type") == t and k not in pc and g_ok(k, t))
        if not pfree or not gfree:
            continue
        cost = np.array([[_d(_ctr(pc[a]["bbox"]), _ctr(gc[b]["bbox"]))
                          for b in gfree] for a in pfree], dtype=float)
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            ren[pfree[i]] = gfree[j]
    if not ren:
        return pred
    pred_pins = pred.get("pins") or {}
    nets = pred.get("nets") or {}
    if cfg.token_align:
        nets = _retoken_nets(nets, pc, gc, pred_pins, gt.get("pins") or {}, ren, cfg)
    return {"components": {ren.get(k, k): v for k, v in pc.items()},
            "pins": {ren.get(k, k): v for k, v in pred_pins.items()},
            "nets": nets}


def _text_field(ctype: str) -> str | None:
    if ctype in _TEXT_NAME:
        return "name"
    if ctype in _TEXT_VALUE:
        return "value"
    return None


def _text_ok(pred_c, gt_c) -> bool:
    f = _text_field(gt_c.get("type", ""))
    if f is None:
        return True
    g = comp_text(gt_c, f)                 # 兼容官方大写 `Name`
    if g is None or (isinstance(g, str) and not g.strip()):
        return True                        # GT null -> 跳过（同 pinname 策略）
    return comp_text(pred_c, f) == g        # 严格字符串


# ---- 四类度量 ------------------------------------------------------------
def component_score(pred: dict, gt: dict, cfg: EvalConfig):
    pc, gc = pred["components"], gt["components"]
    tp = fp = fn = 0
    buckets = {"key_miss": [], "extra": [], "wrong_type": [], "wrong_bbox": [], "wrong_text": []}
    for key in set(pc) | set(gc):
        if key not in pc:                                   # GT 有 pred 无
            fn += 1
            buckets["key_miss"].append(key)
        elif key not in gc:                                 # pred 多出
            fp += 1
            buckets["extra"].append(key)
        else:
            p, g = pc[key], gc[key]
            ok_type = _type_ok(p.get("type"), g.get("type"), cfg)
            ok_box = _bbox_ok(p.get("bbox"), g.get("bbox"), cfg.bbox_tol)
            ok_text = _text_ok(p, g)
            if ok_type and ok_box and ok_text:
                tp += 1
            else:
                fp += 1
                fn += 1
                if not ok_type:
                    buckets["wrong_type"].append(key)
                elif not ok_box:
                    buckets["wrong_bbox"].append(key)
                else:
                    buckets["wrong_text"].append(key)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn), "buckets": buckets}


def pin_score(pred: dict, gt: dict, cfg: EvalConfig):
    pc, gc = pred["components"], gt["components"]
    ppins, gpins = pred["pins"], gt["pins"]
    tp = fp = fn = 0
    skip = cfg.pinname_skip_when_gt_null
    for key in set(pc) | set(gc):
        pg = ppins.get(key, {})
        gg = gpins.get(key, {})
        if key not in gc:
            fp += len(pg)                                   # 多余元件全 FP
            continue
        if key not in pc:
            fn += len(gg)                                   # 缺失元件全 FN
            continue
        gitems = list(gg.items())
        pitems = list(pg.items())
        if not gitems or not pitems:
            fp += len(pitems)
            fn += len(gitems)
            continue
        costs = [[0.0 if _d(_pt(g["point"]), _pt(p["point"])) <= cfg.point_tol else 1.0
                  for _pi, p in pitems] for _gi, g in gitems]
        pairs, _n = _max_cardinality(costs)
        matched_p = [False] * len(pitems)
        matched_g = [False] * len(gitems)
        for gi, pi in pairs:
            matched_g[gi] = matched_p[pi] = True
            g = gitems[gi][1]
            p = pitems[pi][1]
            # 官方：Pin 先按 component key 匹配，再按 pin number 匹配；判定含 pin name/number/坐标。
            # number 即 pin key（pin_A5 允许字母编号），pred 与 GT 必须同 key 才算对——
            # 否则"编号写错但坐标对"会漏判（旧实现只查 pinname）。
            number_ok = pitems[pi][0] == gitems[gi][0]
            gname = g.get("pinname")
            name_ok = (gname is None or (skip and not str(gname).strip())) or p.get("pinname") == gname
            if name_ok and number_ok:
                tp += 1
            else:
                fp += 1
                fn += 1
        fp += sum(1 for i, ok in enumerate(matched_p) if not ok)
        fn += sum(1 for i, ok in enumerate(matched_g) if not ok)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn)}


def net_hypergraph_score(pred: dict, gt: dict):
    """hyperGraph → token 集合，pred/gt net 1:1 匈牙利（cost=-交集），交集>0 配对。"""
    def build(nets):
        return [(nm, frozenset(_hyper_tokens(n["hyperGraph"]))) for nm, n in nets.items()]
    pn = build(pred["nets"])
    gn = build(gt["nets"])
    if not pn or not gn:
        tp = fp = fn = 0
        fp += sum(len(s) for _, s in pn)
        fn += sum(len(s) for _, s in gn)
        return {"tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn), "matched": []}
    costs = np.array([[-len(s1 & s2) for _, s2 in gn] for _, s1 in pn], dtype=float)
    r, c = linear_sum_assignment(costs)
    matched = []
    for i, j in zip(r, c):
        inter = len(pn[i][1] & gn[j][1])
        if inter > 0:
            matched.append((i, j))
    # 配对 net 内：TP=交集；pred 独有 token=FP、gt 独有 token=FN（部分缺失/多报都要扣分，
    # 否则 GT=(A,B) vs pred=(A) 会被算成 F1=1.0——旧实现只对"未配对 net"计 FP/FN，漏了配对内差值）。
    tp = fp = fn = 0
    for i, j in matched:
        pset, gset = pn[i][1], gn[j][1]
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    matched_p = {i for i, _ in matched}
    matched_g = {j for _, j in matched}
    fp += sum(len(s) for k, (_, s) in enumerate(pn) if k not in matched_p)
    fn += sum(len(s) for k, (_, s) in enumerate(gn) if k not in matched_g)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn),
            "matched": matched}


def line_score(pred: dict, gt: dict, netmatch, cfg: EvalConfig):
    """只在 hyperGraph 配对成功的 net 对内比 2 点线段（口径见模块 docstring）。"""
    pn = [(nm, n) for nm, n in pred["nets"].items()]
    gn = [(nm, n) for nm, n in gt["nets"].items()]
    tp = fp = fn = 0
    for i, j in netmatch:
        psegs = [s for s in pn[i][1]["edges"].values()]
        gsegs = [s for s in gn[j][1]["edges"].values()]
        if not psegs or not gsegs:
            fp += len(psegs)
            fn += len(gsegs)
            continue
        costs = [[0.0 if _seg_feasible(ps, gs, cfg.line_tol) else 1.0
                  for gs in gsegs] for ps in psegs]
        pairs, n = _max_cardinality(costs)
        tp += n
        fp += len(psegs) - n
        fn += len(gsegs) - n
    return {"tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn)}


# ---- 汇总 ----------------------------------------------------------------
@dataclass
class EvalResult:
    case_id: str
    comp: dict
    pin: dict
    net: dict
    line: dict
    netmatch: list = field(default_factory=list)

    @property
    def score(self) -> float:
        return 100.0 * (0.3 * self.comp["f1"] + 0.25 * self.pin["f1"]
                        + 0.35 * self.net["f1"] + 0.1 * self.line["f1"])

    def f1s(self) -> dict:
        return {"component": self.comp["f1"], "pin": self.pin["f1"],
                "net": self.net["f1"], "line": self.line["f1"]}

    def to_dict(self) -> dict:
        return {
            "case": self.case_id, "score": round(self.score, 6),
            "f1": {k: round(v, 6) for k, v in self.f1s().items()},
            "component": self.comp, "pin": self.pin,
            "net": {k: v for k, v in self.net.items() if k != "matched"},
            "line": self.line, "net_pairs_matched": len(self.netmatch),
        }

    def render_md(self) -> str:
        s = self.score
        f = self.f1s()
        lines = [
            f"# Evaluate {self.case_id}",
            "",
            f"**Score = {s:.3f}** / 100",
            "",
            "| 方面 | F1 | TP | FP | FN |",
            "|---|---|---|---|---|",
        ]
        for name, r in (("Component", self.comp), ("Pin", self.pin),
                        ("NetHypergraph", self.net), ("NetLine", self.line)):
            lines.append(f"| {name} | {r['f1']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |")
        b = self.comp.get("buckets", {})
        lines += ["", "Component 失败桶：", ""]
        for k, v in b.items():
            if v:
                lines.append(f"- {k}: {v[:20]}{'…' if len(v) > 20 else ''}")
        return "\n".join(lines)


def evaluate(pred: dict, gt: dict, case_id: str = "?", cfg: EvalConfig | None = None) -> EvalResult:
    cfg = cfg or EvalConfig()
    # ⚠ L2 的 key 重挂在这里统一做一次（四类度量共享同一个 pred）⇒ 直接调
    #   `component_score` 之类**不会**自动带上 L2，要自己先 `align_loose`。
    pred = align_loose(pred, gt, cfg)
    comp = component_score(pred, gt, cfg)
    pin = pin_score(pred, gt, cfg)
    net = net_hypergraph_score(pred, gt)
    line = line_score(pred, gt, net["matched"], cfg)
    return EvalResult(case_id=case_id, comp=comp, pin=pin, net=net, line=line,
                      netmatch=net["matched"])


def load_target(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_pred_target(case_id: str, pred: Path) -> Path:
    """兼容 <file>.json / <dir>/<case>.json / <dir>/<case>/<case>.json。"""
    if pred.is_file():
        return pred
    for cand in (pred / f"{case_id}.json", pred / case_id / f"{case_id}.json",
                 pred / f"{case_id}.target.json"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"pred 找不到 {case_id}: {pred}")
