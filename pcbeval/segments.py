# -*- coding: utf-8 -*-
"""wire 折线 -> 2 点线段（target nets.edges 的 edge_i 数据源）。

官方格式要点（prep/work/PROJECT_HANDOFF.md 官方权威格式要点）：
  edge_i 的 value 必须是长度为 2 的 point list [start,end]；
  一条 wire 有多个折点要拆成多条 edge_i，不得写 [p1,p2,p3]。
两端点相同 = 零长度线段（飞线连接点才用）；数据中出现的重复/回走线段逐条保留
（与官方同源数据一致，NetLineF1 的 edges 数量匹配要求我们不能擅自去重）。
"""
from __future__ import annotations


def path_segments(path) -> list[list[list]]:
    """把一条折线（[[x,y],...]）拆成相邻 2 点线段，去零长。

    返回 [ [[x1,y1],[x2,y2]], ... ]；零长度（两端点相同）直接丢弃。
    """
    segs = []
    for a, b in zip(path, path[1:]):
        if a == b:                     # 零长/重复点
            continue
        segs.append([a, b])
    return segs


def wire_segments(wire_paths, flip: bool = False, H: float | None = None) -> list[list]:
    """把一个 edge 的 keyPointPaths.wire（list of 折线）拆成 2 点线段。

    - 每条折线独立拆分（相邻 2 点），跨折线不拼接；
    - 源坐标 y 向下；flip=True 时转成左下角 y 向上（与 pin/bbox 同口径）。
    返回 [ [ {x,y}, {x,y} ], ... ]（净 segment，零长已去除）。
    """
    out = []
    for path in wire_paths or []:
        if not path:
            continue
        for a, b in path_segments(list(path)):
            out.append([_pt(a, flip, H), _pt(b, flip, H)])
    return out


def _pt(p, flip: bool, H: float | None) -> dict:
    x, y = float(p[0]), float(p[1])
    if flip and H is not None:
        y = H - y
    return {"x": x, "y": y}
