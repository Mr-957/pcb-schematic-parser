# -*- coding: utf-8 -*-
"""P1 阶段 0.5.1：target JSON 格式净化器 + 独立校验器（决策 6，保命装置）。

**为什么必须有**：官方评测「单例跑不通 = 该例 0 分」比"精度略低"致命得多；而
`pcbeval.scorer.evaluate` **直接下标取** `pred["components"]/["pins"]/["nets"]`，
缺键即 `KeyError`。现有 `pcbeval/sanity.py` 只校验 `GtBuilder` 自己的产物、
**不校验任意外部 pred**，且缺少 pin 键对齐等检查 —— 本模块补上这个缺口。

**设计原则**
1. `check_target` 才是交付物；`purify` = 「修 → 复检 → 直到稳定（最多 3 轮）」，于是
   "保证合法"成为**可测属性**而非声明。
2. **绝不 clamp 越界 bbox**（默认）。实测官方 GT 自己有 **58 个越界 bbox**
   （44 个 `y1<0`，最大外溢 y=−35px），而 `scorer._bbox_ok` 要求四角都在 GT 的
   `bbox_tol=20px` 内 —— 把 `0003/U5` 的 `y1=−35` clamp 到 0，距离就成了 35 > 20，
   **等于亲手制造一个 `wrong_bbox`**。越界只 warn；`clamp_oob=True` 仅作 opt-in。
3. **不替"合法地没有引脚"的元件造引脚**。空 `pins` 条目 `{}` 是官方合法形态
   （实测 116 处；gnd/v/m3螺丝/block 类常见），必须**保留**而不是补造。
4. **不"修复"官方的正常退化**：`"()"` 空 hyperGraph、单 token net、**零长度 edge
   （实测 95 处，即官方飞线表示）** 一律原样保留。
5. **保真铁律**：`purify(官方 GT) == 官方 GT`（全 210 例）。净化器一旦改写合法输入，
   就是 bug —— 这条测试是第 2 条原则的守门人。

坐标：官方 target 为 **y-up 原点左下**，本模块只查 `[0,W]×[0,H]` 边界，不翻帧。

运行（pcbeval env，项目根）：
  python -m pcbmodel.integrate.purify --audit                # 全 210 例：检 + 保真
  python -m pcbmodel.integrate.purify --case 0025 --check    # 单例校验
  python -m pcbmodel.integrate.purify --pred <p.json> --W 586 --H 435
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.refs import (VOID, VOID_LOOKALIKES, TokenResolver, iter_tokens,
                          make_hypergraph, normalize_void)

TOP_KEYS = ("components", "pins", "nets")
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Issue:
    level: str
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.check}: {self.detail}"


@dataclass
class PurifyReport:
    """三类记录**语义不同，不可混用**：

    - `fixed`    ：无损修复（交换反序 bbox、拆 3 点 edge…）——输出仍代表原意
    - `warnings` ：不必修或修了会变糟（越界 bbox、pinname 缺失…）
    - `dropped`  ：**有损**修复（丢了孤儿键/无法回指的 token…）——**不影响输出合法性**
    - `fatal`    ：复检仍 FAIL —— **只有它决定 `ok`**
    """
    case_id: str = "?"
    fixed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    fatal: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """**输出**是否合法（可交给 scorer）——只看 fatal，不看 dropped。"""
        return not self.fatal

    @property
    def repaired(self) -> bool:
        return bool(self.fixed or self.dropped)

    def to_dict(self) -> dict:
        return {"case": self.case_id, "ok": self.ok, "repaired": self.repaired,
                "fixed": self.fixed, "warnings": self.warnings,
                "dropped": self.dropped, "fatal": self.fatal, "stats": self.stats}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _pt_ok(p) -> bool:
    return isinstance(p, dict) and _is_num(p.get("x")) and _is_num(p.get("y"))


# --------------------------------------------------------------------------- #
# 校验器（纯函数，可作用于任意 dict）
# --------------------------------------------------------------------------- #
def check_target(t, W: float, H: float, case_id: str = "?") -> list[Issue]:
    """校验任意 target dict，返回 Issue 列表。**只报告，不修改。**"""
    out: list[Issue] = []

    def bad(chk, detail):
        out.append(Issue(FAIL, chk, detail))

    def warn(chk, detail):
        out.append(Issue(WARN, chk, detail))

    if not isinstance(t, dict):
        bad("顶层是 dict", f"实际 {type(t).__name__}")
        return out
    for k in TOP_KEYS:
        if k not in t:
            bad("顶层三键齐全", f"缺 {k}")
        elif not isinstance(t[k], dict):
            bad("顶层三键为 dict", f"{k} 是 {type(t[k]).__name__}")
    if any(not isinstance(t.get(k), dict) for k in TOP_KEYS):
        return out

    comps, pins, nets = t["components"], t["pins"], t["nets"]

    # --- components ---
    for d, c in comps.items():
        if not isinstance(c, dict):
            bad("component 是 dict", f"{d!r} 是 {type(c).__name__}")
            continue
        bb = c.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4 and all(_is_num(v) for v in bb)):
            bad("bbox 恰 4 个有限数", f"{d!r} bbox={bb!r}")
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)
        if not (x1 < x2 and y1 < y2):
            bad("bbox 有序非退化", f"{d!r} bbox={bb!r}")
        if not (0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H):
            warn("bbox 越出画布", f"{d!r} bbox={bb!r} 画布 {W}x{H}")
        ty = c.get("type")
        if not (isinstance(ty, str) and ty.strip()):
            warn("type 非空字符串", f"{d!r} type={ty!r}")
        val = c.get("value")
        if val is not None and not isinstance(val, str):
            warn("value 是 str 或 null", f"{d!r} value={val!r}")
        if "Name" not in c and "name" not in c:
            warn("Name 存在", f"{d!r} 无 Name")
        if isinstance(d, str) and d[:1] in VOID_LOOKALIKES:
            bad("∅ 码点正确", f"{d!r} 首位是形近字，应为 U+2205")

    # --- pins ---
    if set(pins) != set(comps):
        miss = sorted(set(comps) - set(pins))
        extra = sorted(set(pins) - set(comps))
        bad("pins 键集与 components 对齐",
            f"缺 {len(miss)} 个（{miss[:3]}）/ 多余 {len(extra)} 个（{extra[:3]}）")
    n_empty = 0
    for d, pd in pins.items():
        if not isinstance(pd, dict):
            bad("pins 条目是 dict", f"{d!r} 是 {type(pd).__name__}")
            continue
        if not pd:
            n_empty += 1                       # 合法：不报
            continue
        for pk, p in pd.items():
            if not (isinstance(pk, str) and pk.startswith("pin_")):
                bad("pin 键以 pin_ 开头", f"{d!r} 的 {pk!r}")
                continue
            if not isinstance(p, dict):
                bad("pin 值是 dict", f"{d!r}.{pk} 是 {type(p).__name__}")
                continue
            if not _pt_ok(p.get("point")):
                bad("pin point 有有限 x/y", f"{d!r}.{pk} point={p.get('point')!r}")
            pn = p.get("pinname")
            if pn is None or (isinstance(pn, str) and not pn.strip()):
                warn("pinname 非空", f"{d!r}.{pk} pinname={pn!r}"
                     "（置空会让 pinF1 恒 0）")
            if pk.startswith("pin_") and pk not in (None,):
                pass

    # --- nets ---
    for nm, net in nets.items():
        if not isinstance(nm, str) or not nm:
            bad("net 键是非空字符串", repr(nm))
        elif nm[:1] == VOID:
            bad("net 键不以 ∅ 开头", f"{nm!r}")
        if not isinstance(net, dict):
            bad("net 值是 dict", f"{nm!r} 是 {type(net).__name__}")
            continue
        hg = net.get("hyperGraph")
        if not isinstance(hg, str):
            bad("hyperGraph 是字符串", f"{nm!r} {type(hg).__name__}")
        else:
            r = TokenResolver(list(comps), pins)
            for tk in iter_tokens(hg):
                got = r.resolve(tk)
                if got is None:
                    bad("hyperGraph token 回指 components", f"{nm!r} token {tk!r}")
                elif got[1] not in (pins.get(got[0]) or {}):
                    bad("hyperGraph token 的 pin 存在",
                        f"{nm!r} token {tk!r} → {got[1]} 不在 pins[{got[0]!r}]")
        edges = net.get("edges")
        if not isinstance(edges, dict):
            bad("edges 是 dict", f"{nm!r} {type(edges).__name__}")
            continue
        for i, (ek, pts) in enumerate(edges.items(), 1):
            if ek != f"edge_{i}":
                bad("edge 键 1-based 连续", f"{nm!r} 第 {i} 个是 {ek!r}")
            if not (isinstance(pts, (list, tuple)) and len(pts) == 2):
                bad("edge 恰 2 点", f"{nm!r}.{ek} len={len(pts) if isinstance(pts, (list, tuple)) else '?'}")
                continue
            for p in pts:
                if not _pt_ok(p):
                    bad("edge 点是 {x,y} 有限数", f"{nm!r}.{ek} {p!r}")
    stats = {"n_comp": len(comps), "n_pin": sum(len(p) for p in pins.values()),
             "n_pin_empty": n_empty, "n_net": len(nets)}
    if not out:
        out.append(Issue(PASS, "全部检查", f"{stats}"))
    return out


# --------------------------------------------------------------------------- #
# 净化器（修 → 复检 → 稳定）
# --------------------------------------------------------------------------- #
def _purify_once(pred, W: float, H: float, rep: PurifyReport, *,
                 clamp_oob: bool, fabricate_missing_pins: bool,
                 drop_dangling: bool) -> dict:
    src = pred if isinstance(pred, dict) else {}

    # 1) 顶层三键
    out: dict = {}
    for k in TOP_KEYS:
        v = src.get(k)
        if not isinstance(v, dict):
            if k in src:
                rep.fixed.append(f"顶层 {k} 非 dict → 置空")
            else:
                rep.fixed.append(f"顶层缺 {k} → 补空")
            v = {}
        out[k] = v
    extra = [k for k in src if k not in TOP_KEYS]
    if extra:
        rep.warnings.append(f"丢弃顶层多余键 {extra}")

    # 2) components
    comps_in = out["components"]
    comps: dict = {}
    dropped: set[str] = set()
    for d, c in comps_in.items():
        if not isinstance(d, str) or not isinstance(c, dict):
            rep.dropped.append(f"component {d!r} 非 dict → 丢弃")
            dropped.add(d)
            continue
        d2 = normalize_void(d)
        if d2 != d:
            rep.fixed.append(f"位号 {d!r} 的 ∅ 形近字 → {d2!r}")
        bb = c.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4 and all(_is_num(v) for v in bb)):
            rep.dropped.append(f"component {d2!r} bbox 非法 {bb!r} → 丢弃")
            dropped.add(d)
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)
        if x1 > x2:
            x1, x2 = x2, x1
            rep.fixed.append(f"{d2!r} bbox x 反序 → 交换")
        if y1 > y2:
            y1, y2 = y2, y1
            rep.fixed.append(f"{d2!r} bbox y 反序 → 交换")
        if x1 == x2:
            x2 = x1 + 0.5
            rep.fixed.append(f"{d2!r} bbox 退化为零宽 → 扩 0.5")
        if y1 == y2:
            y2 = y1 + 0.5
            rep.fixed.append(f"{d2!r} bbox 退化为零高 → 扩 0.5")
        oob = not (0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H)
        if oob:
            if clamp_oob:
                cx1, cy1 = max(0.0, x1), max(0.0, y1)
                cx2, cy2 = min(float(W), x2), min(float(H), y2)
                if not (cx1 < cx2 and cy1 < cy2):
                    rep.dropped.append(f"{d2!r} bbox 越界且 clamp 后退化 → 丢弃")
                    dropped.add(d)
                    continue
                rep.fixed.append(f"{d2!r} bbox 越界 → clamp（⚠ 可能反而偏离官方 GT）")
                x1, y1, x2, y2 = cx1, cy1, cx2, cy2
            else:
                rep.warnings.append(
                    f"{d2!r} bbox 越出画布 {[x1, y1, x2, y2]}（官方 GT 自己也越界，保留不改）")
        ty = c.get("type")
        if not (isinstance(ty, str) and ty.strip()):
            rep.warnings.append(f"{d2!r} type 非法 {ty!r} → 'other'")
            ty = "other"
        name = c.get("Name", c.get("name", d2))
        if not isinstance(name, str):
            name = str(name)
            rep.fixed.append(f"{d2!r} Name 非字符串 → str")
        val = c.get("value")
        if val is not None and not isinstance(val, str):
            rep.warnings.append(f"{d2!r} value 非字符串 {val!r} → 置 null")
            val = None
        comps[d2] = {"Name": name, "type": ty, "value": val,
                     "bbox": [x1, y1, x2, y2]}
        if d2 in comps and d2 != d:
            pass
    if len(comps) != len({k for k in comps_in} - dropped):
        rep.warnings.append("位号归一后出现重名，后者覆盖前者")
    out["components"] = comps

    # 3) pins：键集对齐 + 条目清洗
    # ⚠ 必须与 components 做**同一套** ∅ 归一，否则 `pins["Ø9"]` 会被当成孤儿键删掉，
    # 而 `components["∅9"]` 却已改名 —— 键对齐就此断裂。
    pins_src: dict = {}
    for k, v in out["pins"].items():
        k2 = normalize_void(k) if isinstance(k, str) else k
        if k2 in pins_src and k2 != k:
            rep.warnings.append(f"pins 键 {k!r} 与 {k2!r} 归一后重名 → 后者覆盖前者")
        pins_src[k2] = v
    pins: dict = {}
    for d in comps:
        pd = pins_src.get(d)
        if not isinstance(pd, dict):
            pd = {}
            if d in pins_src:
                rep.fixed.append(f"pins[{d!r}] 非 dict → 置空")
        clean: dict = {}
        for pk, p in pd.items():
            if not (isinstance(pk, str) and pk.startswith("pin_")):
                rep.dropped.append(f"pins[{d!r}] 键 {pk!r} 不以 pin_ 开头 → 丢弃")
                continue
            if not isinstance(p, dict) or not _pt_ok(p.get("point")):
                rep.dropped.append(f"pins[{d!r}].{pk} point 非法 → 丢弃")
                continue
            pn = p.get("pinname")
            if pn is None:
                rep.warnings.append(f"pins[{d!r}].{pk} pinname 缺失 → 置为脚号（会让 pinF1 恒 0）")
                pn = pk[len("pin_"):]
            elif not isinstance(pn, str):
                pn = str(pn)
                rep.fixed.append(f"pins[{d!r}].{pk} pinname 非字符串 → str")
            clean[pk] = {"pinname": pn,
                         "point": {"x": float(p["point"]["x"]),
                                   "y": float(p["point"]["y"])}}
        pins[d] = clean
    for d in pins_src:
        if d not in comps:
            rep.dropped.append(f"pins[{d!r}] 无对应 component → 删除孤儿键")
    out["pins"] = pins
    n_empty = sum(1 for v in pins.values() if not v)

    # 4) nets
    nets_in = out["nets"]
    nets: dict = {}
    n_zerolen = 0
    n_dangling = 0
    n_fabricated = 0
    for i, (nm, net) in enumerate(nets_in.items(), 1):
        name = nm if (isinstance(nm, str) and nm and nm[:1] != VOID) else f"net_{i}"
        if name != nm:
            rep.fixed.append(f"net 键 {nm!r} 非法 → {name!r}")
        while name in nets:
            name += "_2"
        if not isinstance(net, dict):
            rep.dropped.append(f"net {nm!r} 非 dict → 丢弃")
            continue
        resolver = TokenResolver(list(comps), pins)
        toks: list[str] = []
        for tk in iter_tokens(net.get("hyperGraph", "")):
            got = resolver.resolve(tk)
            if got is None:
                n_dangling += 1
                if drop_dangling:
                    rep.dropped.append(f"net {name!r} token {tk!r} 无法回指 → 丢弃该 token")
                else:
                    toks.append(tk)
                continue
            desig, pk = got
            if pk not in pins[desig]:
                if fabricate_missing_pins:
                    # 取舍：token 在**已配对**的 net 里值 0.35/|tokens|；多造一个 pin 只在
                    # 该元件 key 命中时才扣 0.25/|pins|。netF1 权重更大且 |tokens| 很小
                    # （中位 2），故补造弱占优。坐标无从得知，置 (0,0)。
                    pins[desig][pk] = {"pinname": pk[len("pin_"):],
                                       "point": {"x": 0.0, "y": 0.0}}
                    n_fabricated += 1
                    rep.warnings.append(f"net {name!r} token {tk!r} 的 pin 不存在 → 补造")
                else:
                    n_dangling += 1
                    rep.dropped.append(f"net {name!r} token {tk!r} 的 pin 不存在 → 丢弃")
                    continue
            toks.append(tk)
        # 重建 hyperGraph（顺带消掉逗号后空格这类问题）
        hg = make_hypergraph(toks)
        edges_in = net.get("edges")
        edges: dict = {}
        if isinstance(edges_in, dict):
            for pts in edges_in.values():
                if not (isinstance(pts, (list, tuple)) and len(pts) >= 1):
                    rep.dropped.append(f"net {name!r} 有空 edge → 丢弃该 edge")
                    continue
                good = [p for p in pts if _pt_ok(p)]
                if not good:
                    rep.dropped.append(f"net {name!r} edge 全是非法点 → 丢弃该 edge")
                    continue
                if len(good) == 1:
                    good = [good[0], dict(good[0])]      # 单点 → 零长度段（官方允许）
                    rep.fixed.append(f"net {name!r} 单点 edge → 补成零长度段")
                for a, b in zip(good, good[1:]):          # >2 点 → 连续拆对（官方规则）
                    edges[f"edge_{len(edges) + 1}"] = [
                        {"x": float(a["x"]), "y": float(a["y"])},
                        {"x": float(b["x"]), "y": float(b["y"])}]
                else:
                    pass
        elif edges_in is not None:
            rep.fixed.append(f"net {name!r} edges 非 dict → 置空")
        for pts in edges.values():
            if pts[0]["x"] == pts[1]["x"] and pts[0]["y"] == pts[1]["y"]:
                n_zerolen += 1
        nets[name] = {"hyperGraph": hg, "edges": edges}
    out["nets"] = nets

    rep.stats = {"n_comp": len(comps), "n_pin": sum(len(v) for v in pins.values()),
                 "n_pin_empty": n_empty, "n_net": len(nets),
                 "n_zero_len_edge": n_zerolen, "n_dangling_dropped": n_dangling,
                 "n_pin_fabricated": n_fabricated}
    return out


def purify(pred, *, W: float, H: float, case_id: str = "?", clamp_oob: bool = False,
           fabricate_missing_pins: bool = True, drop_dangling: bool = True,
           max_passes: int = 3) -> tuple[dict, PurifyReport]:
    """把任意 pred 修成**保证合法**的 target。返回 `(target, report)`。"""
    rep = PurifyReport(case_id=case_id)
    cur = pred
    for _ in range(max(1, max_passes)):
        n_fix_before = len(rep.fixed)
        cur = _purify_once(cur, W, H, rep, clamp_oob=clamp_oob,
                           fabricate_missing_pins=fabricate_missing_pins,
                           drop_dangling=drop_dangling)
        if len(rep.fixed) == n_fix_before:
            break                       # 已稳定
    # 复检
    issues = check_target(cur, W, H, case_id)
    for it in issues:
        if it.level == FAIL:
            rep.fatal.append(f"复检未过 {it.check}: {it.detail}")
        elif it.level == WARN:
            rep.warnings.append(f"{it.check}: {it.detail}")
    return cur, rep


def render_md(rep: PurifyReport) -> str:
    L = [f"# 净化报告 {rep.case_id}", "",
         f"- 合法（无 FAIL）：**{rep.ok}**　已修：{rep.repaired}",
         f"- 统计：`{rep.stats}`", ""]
    for title, items in (("修复", rep.fixed), ("告警", rep.warnings),
                         ("有损丢弃", rep.dropped), ("致命（输出不合法）", rep.fatal)):
        L.append(f"## {title}（{len(items)}）")
        L += [f"- {x}" for x in items[:50]] or ["- （无）"]
        if len(items) > 50:
            L.append(f"- … 另有 {len(items) - 50} 条")
        L.append("")
    return "\n".join(L)


def _audit(show: int) -> int:
    from pcbeval.refs import _official_targets
    from PIL import Image

    n = n_faithful = n_ok = 0
    bad_fidelity: list[str] = []
    bad_check: list[str] = []
    for cid, t in _official_targets():
        # 找同目录 PNG 取 W,H
        W = H = None
        import glob
        import os
        for root in (f"赛题六公开数据集/200_train_cases/{cid}",
                     *glob.glob(f"赛题六公开数据集/10_GTcase/{cid}-*")):
            for p in glob.glob(os.path.join(root, "*.png")):
                W, H = Image.open(p).size
                break
            if W:
                break
        if not W:
            continue
        n += 1
        issues = check_target(t, W, H, cid)
        fails = [i for i in issues if i.level == FAIL]
        if fails:
            bad_check.append(f"{cid}: {fails[0]}")
        out, rep = purify(t, W=W, H=H, case_id=cid)
        if rep.ok:
            n_ok += 1
        if out == t:
            n_faithful += 1
        else:
            diff = [k for k in TOP_KEYS if out.get(k) != t.get(k)]
            bad_fidelity.append(f"{cid}: 改动了 {diff}")
    print(f"[purify] 审计 {n} 例")
    print(f"  check_target 0 FAIL : {n - len(bad_check)}/{n}")
    print(f"  purify 后合法        : {n_ok}/{n}")
    print(f"  **保真（逐字不变）** : {n_faithful}/{n}")
    for x in bad_check[:show]:
        print(f"    CHECK-FAIL {x}")
    for x in bad_fidelity[:show]:
        print(f"    NON-FAITHFUL {x}")
    ok = (n > 0 and not bad_check and not bad_fidelity and n_ok == n)
    print("[purify] AUDIT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="purify", description="target JSON 格式净化器/校验器")
    ap.add_argument("--audit", action="store_true", help="全 210 例：校验 + 保真")
    ap.add_argument("--case", default=None, help="官方 case id（配合 --check）")
    ap.add_argument("--pred", default=None, help="外部 pred JSON")
    ap.add_argument("--W", type=float, default=None)
    ap.add_argument("--H", type=float, default=None)
    ap.add_argument("--check", action="store_true", help="只校验不修")
    ap.add_argument("--clamp-oob", action="store_true", help="越界 bbox 做 clamp（默认只告警）")
    ap.add_argument("--out", default=None, help="净化后写盘路径")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args(argv)

    if a.audit:
        return _audit(a.show)

    if a.case:
        from pcbeval.refs import _official_targets
        from PIL import Image
        import glob
        import os
        for cid, t in _official_targets():
            if cid != a.case:
                continue
            W = H = None
            for root in (f"赛题六公开数据集/200_train_cases/{cid}",
                         *glob.glob(f"赛题六公开数据集/10_GTcase/{cid}-*")):
                g = glob.glob(os.path.join(root, "*.png"))
                if g:
                    W, H = Image.open(g[0]).size
                    break
            if a.check:
                for it in check_target(t, W, H, cid):
                    print(it)
                return 0
            out, rep = purify(t, W=W or 1, H=H or 1, case_id=cid, clamp_oob=a.clamp_oob)
            print(render_md(rep))
            return 0 if rep.ok else 1
        print(f"[purify] 找不到 case {a.case}")
        return 1

    if a.pred:
        p = Path(a.pred)
        if a.W is None or a.H is None:
            print("[purify] 外部 pred 必须给 --W/--H（target 里没有画布尺寸）")
            return 2
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if a.check:
            for it in check_target(raw, a.W, a.H, p.stem):
                print(it)
            return 0
        out, rep = purify(raw, W=a.W, H=a.H, case_id=p.stem, clamp_oob=a.clamp_oob)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
        print(render_md(rep))
        return 0 if rep.ok else 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
