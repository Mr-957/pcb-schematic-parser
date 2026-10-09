# -*- coding: utf-8 -*-
"""net 分组（Oracle A = .net 注释块）+ segments -> target nets。

数据权威性（见 prep/work/gt_format_report.txt）：
  .net 里 `* E_<k> = <netname> : P_<p>(C_<c>) ...` 是官方流水线推导的权威
  net 分组。同 netname 的多个 edge 行合并成一个 net（netlabel 同名飞线、
  Gnd→"0" 语义都已编码其中）；netlabel 为参数值形态(∅)时净名回退 NET_E_<k>，
  原注释附带的说明括号须剥除，只留 NET_E_<k>。

edge_i 来源：该 net 下每条 edge 的 keyPointPaths.wire 折线 -> 2 点线段
（pcbeval.segments.wire_segments），逐条忠实保留（见 segments.py 注释）。

坐标：与 pin/bbox 同口径 —— gtbuilder 内部源坐标(y 向下)，此处按 cfg.flip_y
统一翻转（本地默认 False）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .segments import wire_segments

_HEADER_RE = re.compile(r"^\*\s*(E_\w+)\s*=\s*(.+)$")
_PIN_RE = re.compile(r"\b(P_\w+)\s*\(\s*(C_\w+)\s*\)")
_TAIL_SEP_RE = re.compile(r"\s*:\s*$")


@dataclass
class NetLine:
    """一条 `* E_k = <net> : P_x(C_y)...` 注释（eid 对应的 edge）。"""
    eid: str
    netname: str            # 已净名（∅ 回退名去掉了说明括号）
    pins: list = field(default_factory=list)   # [(pid, cid), ...]，.net 行内顺序


def clean_netname(eid: str, raw: str) -> str:
    """回退名剥注解：`NET_E_4（原 netlabel "∅52" 为参数值...）` -> `NET_E_4`。

    用 \b 防误伤：E_1 的 NET_E_1 不会匹配到 NET_E_10 的头部。
    """
    m = re.match(r"(NET_" + re.escape(eid) + r")\b", raw)
    return m.group(1) if m else raw.strip()


def parse_net_comments(text: str) -> list[NetLine]:
    """解析 .net 全部 `* E_k = <net> : pins` 注释行（逐行，跨 300 例验证）。"""
    out: list[NetLine] = []
    for raw in text.splitlines():
        m = _HEADER_RE.match(raw.strip())
        if not m:
            continue
        eid, rest = m.group(1), m.group(2).strip()
        pins = _PIN_RE.findall(rest)
        name = _PIN_RE.sub("", rest)
        name = _TAIL_SEP_RE.sub("", name).strip()
        out.append(NetLine(eid=eid, netname=clean_netname(eid, name), pins=pins))
    return out


def aggregate(cfg, cv: dict, comps: list, net_lines: list[NetLine]):
    """按 netname 聚合成 target nets dict + 供 sanity 的 meta。

    comps：GtBuilder 已分配好位号/pin 的 CompRec 列表（pid 反查
    (designator, num) 的唯一事实来源）。

    返回 (nets, meta)：
      nets: {netname: {"hyperGraph": str, "edges": {edge_i: [pt,pt], ...}}}
      meta: {pid_net, net_pins, dangling_pids, unannotated_edges, empty_nets,
             net_segcount, order}（供 sanity / 报告，不属 target 输出）
    """
    # pid -> (designator, pin_num)
    rev: dict[str, tuple[str, int]] = {}
    for c in comps:
        for p in c.pins:
            rev[p.pid] = (c.designator, p.num)

    H = cv["canvas"]["height"]
    flip = bool(cfg.flip_y)

    net_order: list[str] = []
    net_pins: dict[str, list[str]] = {}        # netname -> 唯一 pid 列表(源顺序)
    net_lines_by: dict[str, list[NetLine]] = {}  # netname -> 组成行
    for nl in net_lines:
        d = net_pins.setdefault(nl.netname, [])
        net_lines_by.setdefault(nl.netname, []).append(nl)
        if nl.netname not in net_order:
            net_order.append(nl.netname)
        for pid, _cid in nl.pins:
            if pid not in d:
                d.append(pid)

    dangling: list[tuple[str, str]] = []        # (netname, pid)
    nets: dict[str, dict] = {}
    segcount: dict[str, int] = {}
    for nm in net_order:
        pids = net_pins[nm]
        toks = []
        for pid in pids:
            hit = rev.get(pid)
            if hit is None:
                dangling.append((nm, pid))
                continue
            desig, num = hit
            toks.append(f"{desig}.{num}")
        hyper = "(" + ",".join(toks) + ")"

        # edge_i 编号：跨组成行按源顺序累计（同 net 的 0 会跨 E_5+E_10 编 1..N）
        edges: dict[str, list] = {}
        i = 1
        for nl in net_lines_by[nm]:
            e = cv["edges"].get(nl.eid)
            if e is None:
                continue
            segs = wire_segments(e.get("keyPointPaths", {}).get("wire"),
                                 flip=flip, H=H)
            for s in segs:
                edges[f"edge_{i}"] = s
                i += 1
        segcount[nm] = i - 1
        nets[nm] = {"hyperGraph": hyper, "edges": edges}

    # ---- meta（sanity 用，不属于 target）----
    pid_net: dict[str, str] = {}
    for nm in net_order:
        for pid in net_pins[nm]:
            pid_net.setdefault(pid, nm)         # 冲突也算（多重归属 sanity 查）
    annotated_eids = {nl.eid for nl in net_lines}
    unannotated = [
        eid for eid, e in cv["edges"].items()
        if e.get("connect") and eid not in annotated_eids
    ]
    empty_nets = [nm for nm in net_order if not net_pins[nm]]

    meta = {
        "pid_net": pid_net,
        "net_pins": net_pins,
        "net_order": net_order,
        "net_segcount": segcount,
        "dangling_pids": dangling,
        "unannotated_edges": unannotated,
        "empty_nets": empty_nets,
        "annotated_eids": annotated_eids,
    }
    return nets, meta
