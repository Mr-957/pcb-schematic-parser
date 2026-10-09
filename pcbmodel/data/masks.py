# -*- coding: utf-8 -*-
"""P1 阶段 0.3：U-Net 二值导线掩码（y-up 0/1 图）。

**训练用真值 = 官方 `nets.edges`**（决策 15，2026-09-09 决定；此前"源 edges 全量"已作废）
→ 见 `rasterize_target_wire_mask`。`rasterize_wire_mask`（源 CVJsonStd）**仅作对照/预训练**，
不再是 U-Net 的真值来源。

⚠ 官方 edges 的两条性质（指南附录一 §7，2026-09-10 补录）：
  * **edges 数量参与评分**（"多余或缺少都会扣分"）→ 多折点必须拆成 2 点段；
  * **飞线连接点是零长度线段（起点 == 终点）**——官方 200 例实测 95/31749（0.30%）。
    PIL 对零长度 `line` 会画 1 个像素，所以掩码里表现为一个点，**不要当噪声滤掉**。

来源字段（Agent 已实测 0119/0001/0004）：每个 `edges[E_x].keyPointPaths.wire` 是
折线列表（每条折线 = 相邻点连线）；`edges[E_x].coordinates` 是整条细线像素中心线
（keyPointPaths 缺失时的兜底）。坐标均为源帧 y-down。

约定（决策 2）：**先在 y-down 原生帧光栅化，收尾一次 flipud** 成 canonical y-up——
只在本模块与 loader 两处出现翻帧，其余模块只消费 y-up。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根


def source_wire_paths(edge: dict) -> list[list]:
    """取一个 source edge 的全部可见导线折线（y-down 像素坐标）。

    优先 keyPointPaths.wire；缺失/为空时兜底 coordinates（细线像素中心线）。
    返回 list of polyline（每条 = [[x,y], ...]）。
    """
    paths = (edge.get("keyPointPaths") or {}).get("wire") or []
    out = [list(p) for p in paths if p]
    if not out:
        coords = edge.get("coordinates")
        if coords:
            out = [list(coords)]
    return out


def wire_segments_bounds(segments: list[list], W: int, H: int) -> int:
    """越出画布 (0..W) × (0..H) 的 2 点线段条数（含端点整数舍入误差容 0.5px）。"""
    bad = 0
    for seg in segments:
        for x, y in (seg[0], seg[1]):
            if not (-0.5 <= float(x) <= W + 0.5 and -0.5 <= float(y) <= H + 0.5):
                bad += 1
                break
    return bad


def rasterize_wire_mask(cv: dict, flip_up: bool = True,
                        width: int = 2) -> np.ndarray:
    """源 CVJsonStd -> 二值导线掩码（uint8 0/1，flip_up=True 时返回 canonical y-up）。

    - 画到 y-down 原生帧的 PIL 'L' 图（越界坐标被 PIL 自动裁剪，不报错）；
    - flip_up=True 收尾 np.flipud（决策 2：先 y-down 光栅化、后统一翻转）。
    """
    W = int(cv["canvas"]["width"])
    H = int(cv["canvas"]["height"])
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for edge in cv.get("edges", {}).values():
        for path in source_wire_paths(edge):
            pts = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in path]
            if len(pts) >= 2:
                dr.line(pts, fill=255, width=width)
    arr = (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)
    return np.flipud(arr) if flip_up else arr


def rasterize_target_wire_mask(target: dict, W: int, H: int,
                               width: int = 2) -> np.ndarray:
    """**官方 target** -> 二值导线掩码（决策 15：真值 = 官方 nets.edges，uint8 0/1 y-up）。

    官方 target 坐标已是 y-up，与 P0 源帧相反：先在"y-down 栅格"里按 y_pil = H − y_up
    画线，再 flipud 回 y-up 索引（与 `rasterize_wire_mask` 同一约定：arr[y, x] 对应
    y-up 坐标 y）。官方 edges 只含 net 内导线（每条恰 2 点）。
    """
    img = Image.new("L", (int(W), int(H)), 0)
    dr = ImageDraw.Draw(img)
    for net in target.get("nets", {}).values():
        for seg in net.get("edges", {}).values():
            if len(seg) != 2:
                continue
            pts = [(int(round(float(p["x"]))), int(round(H - float(p["y"]))))
                   for p in seg]
            dr.line(pts, fill=255, width=width)
    arr = (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)
    return np.flipud(arr)


def target_mask_stats(target: dict) -> dict:
    """官方 target 的掩码可生成性自查：net 数 / 线段数 / 非 2 点段。"""
    n_nets = len(target.get("nets", {}))
    n_segs = 0
    n_bad = 0
    for net in target.get("nets", {}).values():
        for seg in net.get("edges", {}).values():
            n_segs += 1
            if len(seg) != 2:
                n_bad += 1
    return {"nets": n_nets, "segs": n_segs, "not_two_point": n_bad}


def mask_stats(cv: dict) -> dict:
    """可生成性自查：每例源 edges 数 / 折线段(相邻2点)数 / 越界段计数。"""
    W = int(cv["canvas"]["width"])
    H = int(cv["canvas"]["height"])
    n_edges = len(cv.get("edges", {}))
    n_paths = 0
    n_segs = 0
    n_oob = 0
    for edge in cv.get("edges", {}).values():
        for path in source_wire_paths(edge):
            n_paths += 1
            segs = [[path[i], path[i + 1]] for i in range(len(path) - 1)
                    if list(path[i]) != list(path[i + 1])]
            n_segs += len(segs)
            n_oob += wire_segments_bounds(segs, W, H)
    return {"src_edges": n_edges, "paths": n_paths, "segs": n_segs,
            "out_of_bounds_segs": n_oob}


if __name__ == "__main__":
    # 自检：光栅化掩码维度 == canvas、flip 前后互为上下翻转
    import json
    if len(sys.argv) < 2:
        print("用法: python pcbmodel/data/masks.py <case_id>")
        sys.exit(2)
    from pcbeval.caseio import discover_case_files, load_cvjsonstd
    from pcbeval.config import EvalConfig
    cid = sys.argv[1]
    cv = load_cvjsonstd(discover_case_files(cid, EvalConfig()).cvjson)
    up = rasterize_wire_mask(cv, flip_up=True)
    down = rasterize_wire_mask(cv, flip_up=False)
    assert up.shape == (cv["canvas"]["height"], cv["canvas"]["width"])
    assert np.array_equal(up, np.flipud(down)), "flip 帧不一致"
    st = mask_stats(cv)
    print(f"[masks] {cid} canvas={cv['canvas']} mask.sum={int(up.sum())} "
          f"{st} (oob={st['out_of_bounds_segs']}) flip_check=PASS")
