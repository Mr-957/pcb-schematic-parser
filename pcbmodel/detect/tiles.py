# -*- coding: utf-8 -*-
"""P1 阶段 1.1：大图切块 + 跨缝合并（决策 19；用户已选「按原分辨率切块」）。

**为什么必须切块**：元件框**中位仅 20×19px**（p25 15/10、p90 34/33），而图像中位
1.0MPix、**最大 17.4MPix（5100×3509）、14 例宽 >2000px**。把 5100px 的图直接 resize 到
YOLO 默认 1280，20px 的元件被压成 **5px** —— 低于任何合理 backbone 的 stride。

**为什么"按原分辨率切块"而不是 resize**：元件**绝对像素尺寸跨案例高度一致**
（宽 p25/中位/p75 = 15/20/20）。在原分辨率下切块 = **把目标尺度归一化**，
一套 anchor、一个尺度，不需要多尺度体操。这是本方案最关键的收益。

**坐标系（⚠ 与项目其余部分不同，务必看清）**
    本模块**一律用 y-down（图像原生）像素坐标**，理由：
      - 切块产物要**保持文字正立** —— 阶段 1.2 的 OCR 要用这些裁剪图；y-up 会把文字上下翻转
      - 目检叠加图时不会左右为难
    官方 GT 是 **y-up**，进入本模块前必须用 `loader.bbox_flip_down()` 转成 y-down，
    出模块（写 target JSON）时再转回 y-up。**转换点只有这一处，不要散布。**

运行（pcbeval env，项目根）：
  python -m pcbmodel.detect.tiles --audit          # 全 210 例：块数 / 覆盖率审计
  python -m pcbmodel.detect.tiles --case 0003      # 单例切块明细
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.caseio import discover_official
from pcbeval.config import OFFICIAL_GT_ROOT, OFFICIAL_TRAIN_ROOT

# 实测依据：元件最大边 p99 = 309、p99.9 = 749、绝对最大 1322；>512px 的只有 32 个。
# overlap 512 可**完整容纳 99.66%** 的元件（重叠 ≥ 最大目标边长是"每个目标至少完整落进一块"的充分条件）。
DEFAULT_TILE = 1280      # = YOLO imgsz → 裁剪图 1:1 喂进去，不再重采样
DEFAULT_STRIDE = 768     # overlap = 1280 - 768 = 512


@dataclass(frozen=True)
class TilePlan:
    tile: int = DEFAULT_TILE
    stride: int = DEFAULT_STRIDE

    @property
    def overlap(self) -> int:
        return self.tile - self.stride


def _axis_starts(total: int, tile: int, stride: int) -> list[int]:
    """一个轴上所有切块起点，保证**并集恰好覆盖** `[0, total]`。"""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)          # 末块贴边，避免尾部漏掉
    return starts


def plan_tiles(W: int, H: int, plan: TilePlan | None = None) -> list[tuple[int, int, int, int]]:
    """→ `[(x0, y0, x1, y1), ...]`（y-down 像素），并集覆盖整图。"""
    plan = plan or TilePlan()
    xs = _axis_starts(W, plan.tile, plan.stride)
    ys = _axis_starts(H, plan.tile, plan.stride)
    return [(x, y, min(x + plan.tile, W), min(y + plan.tile, H))
            for y in ys for x in xs]


def clip_box(box, rect, *, min_side: float = 2.0):
    """把 box 裁到 rect 内。可见边长 < `min_side` 时返回 None（碎条不生成标签）。"""
    x0, y0, x1, y1 = rect
    a1, b1, a2, b2 = box
    cx1, cy1 = max(a1, x0), max(b1, y0)
    cx2, cy2 = min(a2, x1), min(b2, y1)
    if cx2 - cx1 < min_side or cy2 - cy1 < min_side:
        return None
    return [cx1, cy1, cx2, cy2]


def visible_ratio(box, rect) -> float:
    got = clip_box(box, rect, min_side=0.0)
    if got is None:
        return 0.0
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    return ((got[2] - got[0]) * (got[3] - got[1])) / area


def assign_box(box, tiles, *, min_visible: float = 0.5, min_side: float = 2.0) -> list[int]:
    """box 应写进哪些块的标签。

    判据：`可见面积比 ≥ min_visible` **或** 裁剪后任一边 ≥ 16px。
    后者保证"大目标被缝切开"时两半都会带标签（再由合并阶段去重），
    而不是一边有框一边没有。
    """
    out: list[int] = []
    for i, rect in enumerate(tiles):
        got = clip_box(box, rect, min_side=min_side)
        if got is None:
            continue
        big = (got[2] - got[0]) >= 16.0 or (got[3] - got[1]) >= 16.0
        if visible_ratio(box, rect) >= min_visible or big:
            out.append(i)
    return out


def to_tile_norm(box, rect):
    """全图 box → 该块内的 YOLO 归一化 `[cx, cy, w, h]`（都在 y-down 下，方向自洽）。"""
    got = clip_box(box, rect, min_side=0.0)
    if got is None:
        return None
    x0, y0, x1, y1 = rect
    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
    cx = (got[0] + got[2]) / 2.0 - x0
    cy = (got[1] + got[3]) / 2.0 - y0
    return [cx / tw, cy / th, (got[2] - got[0]) / tw, (got[3] - got[1]) / th]


def to_full_box(norm, rect):
    """YOLO 归一化 → 全图 y-down box。"""
    x0, y0, x1, y1 = rect
    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
    cx, cy, w, h = norm
    cx, cy, w, h = cx * tw + x0, cy * th + y0, w * tw, h * th
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def iom(a, b) -> float:
    """交并 **min 面积**（containment）—— 用来抓「整框 vs 半框」这种 NMS 抓不到的情况。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / amin if amin > 0 else 0.0


def merge_across_tiles(dets, *, iou_thr: float = 0.5, iom_thr: float = 0.8,
                       class_aware: bool = True):
    """跨缝去重。`dets` = `[{"box":[..], "score":float, "cls":int|str}, ...]`。

    **两步，缺一不可**：
      1. **NMS**（IoU）—— 同一目标在相邻两块里各被完整检出（框几乎重合）
      2. **IoM**（containment）—— 同一目标**一块里是整框、另一块里是半框**，
         此时 IoU 可能只有 ~0.5 而漏过 NMS；IoM = 交/min面积 ≈ 1.0 能抓住
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i].get("score", 1.0))
    keep: list[int] = []
    for i in order:
        bi, ci = dets[i]["box"], dets[i].get("cls")
        drop = False
        for j in keep:
            bj, cj = dets[j]["box"], dets[j].get("cls")
            if class_aware and ci != cj:
                continue
            if iou(bi, bj) >= iou_thr or iom(bi, bj) >= iom_thr:
                drop = True
                break
        if not drop:
            keep.append(i)
    return [dets[i] for i in sorted(keep)]


# --------------------------------------------------------------------------- #
# 审计
# --------------------------------------------------------------------------- #
def _iter_cases(limit: int = 0):
    from PIL import Image
    for root, kind in ((OFFICIAL_TRAIN_ROOT, "train"), (OFFICIAL_GT_ROOT, "gt")):
        if not Path(root).is_dir():
            continue
        for d in sorted(Path(root).iterdir()):
            if not d.is_dir():
                continue
            cid = d.name.split("-", 1)[0] if kind == "gt" else d.name
            oc = discover_official(cid, root, kind=kind)
            if oc.target is None or oc.image is None:
                continue
            with open(oc.target, encoding="utf-8") as f:
                t = json.load(f)
            with Image.open(oc.image) as _im:
                W, H = _im.size
            yield cid, t, W, H
            if limit and cid == str(limit):
                return


def audit(limit: int = 0, plan: TilePlan | None = None) -> dict:
    plan = plan or TilePlan()
    n_cases = n_tiles = n_max = n_boxes = 0
    contained = split_like = 0
    big_boxes = truncated = 0
    trunc_examples: list[dict] = []
    for _cid, t, W, H in _iter_cases(limit):
        tiles = plan_tiles(W, H, plan)
        n_cases += 1
        n_tiles += len(tiles)
        n_max = max(n_max, len(tiles))
        for c in t.get("components", {}).values():
            bb = c["bbox"]
            # GT 是 y-up → 转 y-down 再判包含（本模块统一 y-down）
            x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
            dbox = [x1, H - y2, x2, H - y1]
            n_boxes += 1
            if (dbox[2] - dbox[0]) > plan.stride or (dbox[3] - dbox[1]) > plan.stride:
                big_boxes += 1
                continue
            idx = assign_box(dbox, tiles)
            full = [i for i, r in enumerate(tiles)
                    if clip_box(dbox, r, min_side=0.0) == dbox]
            if full:
                contained += 1
            elif idx:
                split_like += 1
            else:
                # 既非完整容纳、也不生成标签 —— 实测只剩「被画布边缘截断」的极少数
                # （官方 GT 自己有 58 个越界 bbox）。**刻意不标**：把 11px 的残条
                # 当标签会教模型"这个元件只有 11px 高"，而它其实是被裁掉了 19px。
                truncated += 1
                if len(trunc_examples) < 5:
                    trunc_examples.append(
                        {"case": _cid, "ratio_visible": round(
                            max(visible_ratio(dbox, r) for r in tiles), 3)})
    return {"n_cases": n_cases, "n_tiles": n_tiles, "max_tiles_per_case": n_max,
            "n_boxes": n_boxes, "fully_contained": contained,
            "split_but_labelled": split_like, "bigger_than_stride": big_boxes,
            "truncated_no_label": truncated, "truncated_examples": trunc_examples,
            "tile": plan.tile, "stride": plan.stride, "overlap": plan.overlap}


def render_md(r: dict) -> str:
    n = max(r["n_boxes"], 1)
    return "\n".join([
        "# 切块审计（y-down 原分辨率）", "",
        f"> tile={r['tile']}　stride={r['stride']}　overlap={r['overlap']}", "",
        "| 项 | 值 |", "|---|---:|",
        f"| 案例数 | {r['n_cases']} |",
        f"| 切块总数 | {r['n_tiles']}（均值 {r['n_tiles'] / max(r['n_cases'], 1):.1f}/例，"
        f"最多 {r['max_tiles_per_case']}） |",
        f"| 元件框总数 | {r['n_boxes']} |",
        f"| **被某块完整容纳** | {r['fully_contained']}"
        f"（**{100 * r['fully_contained'] / n:.2f}%**） |",
        f"| 跨缝但有标签 | {r['split_but_labelled']}（{100 * r['split_but_labelled'] / n:.2f}%） |",
        f"| 大于 stride（跨多块） | {r['bigger_than_stride']}"
        f"（{100 * r['bigger_than_stride'] / n:.2f}%） |",
        f"| 画布边缘截断、**刻意不标** | {r.get('truncated_no_label', 0)}"
        f"（{100 * r.get('truncated_no_label', 0) / n:.2f}%） |", "",
        "> 「被完整容纳」是理想情况（该目标在某块里是整框）；「跨缝但有标签」由合并阶段处理。",
        "> 大于 stride 的目标**没有任何块能完整容纳** —— 它们靠 `assign_box` 的 16px 规则"
        "带标签，是切片方案的已知上限。",
        "> 「截断不标」是**刻意**的：把 11px 残条当标签会教模型\"这个元件只有 11px 高\"，"
        "而它其实是被画布裁掉了。官方 GT 自己有 58 个越界 bbox，故这属于**合法输入**。", ""])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="tiles", description="大图切块 / 标签归属 / 跨缝合并")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--case", default=None)
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "detect"))
    a = ap.parse_args(argv)
    plan = TilePlan(a.tile, a.stride)

    if a.case:
        for cid, t, W, H in _iter_cases():
            if cid != a.case:
                continue
            tiles = plan_tiles(W, H, plan)
            print(f"[tiles] {cid} {W}x{H} -> {len(tiles)} 块  (tile={plan.tile} stride={plan.stride})")
            for i, r in enumerate(tiles):
                print(f"   #{i:<3d} {r}")
            for d, c in list(t["components"].items())[:8]:
                x1, y1, x2, y2 = c["bbox"]
                dbox = [x1, H - y2, x2, H - y1]
                print(f"   {d:10s} ydown={[round(v, 1) for v in dbox]} -> tiles {assign_box(dbox, tiles)}")
            return 0
        print(f"[tiles] 找不到 case {a.case}")
        return 1

    if a.audit:
        r = audit(a.limit, plan)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tiling_audit.md").write_text(render_md(r), encoding="utf-8")
        (out / "tiling_audit.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[tiles] {r['n_cases']} 例 -> {r['n_tiles']} 块（均值 "
              f"{r['n_tiles'] / max(r['n_cases'], 1):.1f}，最多 {r['max_tiles_per_case']}）")
        n = max(r["n_boxes"], 1)
        print(f"   完整容纳 {r['fully_contained']}/{r['n_boxes']} = "
              f"{100 * r['fully_contained'] / n:.2f}%")
        print(f"   跨缝有标签 {r['split_but_labelled']} ｜ 大于 stride {r['bigger_than_stride']}")
        print(f"   -> {out / 'tiling_audit.md'}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
