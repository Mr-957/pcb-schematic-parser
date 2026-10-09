# -*- coding: utf-8 -*-
"""P1 阶段 1.1：**框中位误差** 评估 —— 决定引脚几何模块（A5）能否存活。

**为什么不能只看 mAP**：`pcbmodel/integrate/pin_geom.py` 已实测（GT 框输入下的 jitter 曲线）：

| 框 jitter | 推导点 ≤5px 命中 | pinF1 |
|---|---:|---:|
| 0 | 94.4% | **0.4594** |
| ±2px | 92.4% | 0.4517 |
| ±5px | 82.0% | 0.3621 |
| ±10px | 26.2% | **0.1033** |

引脚坐标**整份继承框边缘误差（1:1）**，而引脚容差只有 **5px**、元件容差是 20px。
所以 mAP 高不代表 A5 可用 —— **必须直接量框误差，并直接量由此产生的引脚误差**。

**本脚本做两件事**（不查表、直接端到端量）：
  1. 预测框 vs GT 框的**逐边误差**中位/p90
  2. 用**预测框**跑 `pin_geom.derive_2pin` → 与 GT 引脚点比 → **引脚命中率与 pinF1**

坐标：本模块在 **y-down** 下推理与比对（与 `tiles.py` 一致）。

运行（pcbmodel env，需 torch + ultralytics）：
  <env>/bin/python -m pcbmodel.detect.eval_box --weights <run>/weights/best.pt --split val
  <env>/bin/python -m pcbmodel.detect.eval_box --weights ... --cases 0025,0119 --limit 5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.config import OFFICIAL_GT_ROOT, OFFICIAL_TRAIN_ROOT
from pcbmodel.detect.tiles import (DEFAULT_STRIDE, DEFAULT_TILE, TilePlan, merge_across_tiles,
                                   plan_tiles, to_full_box)
from pcbmodel.integrate.pin_geom import DERIVABLE_2PIN, PIN1_TRUSTWORTHY, derive_2pin

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "detect"
POINT_TOL = 5.0
VOID = "∅"


def _to_up_box(box_ydown, H: float):
    """y-down 框 → y-up 框。**调用 `pin_geom` 前必须转**（见 `pin_error` 的警示）。"""
    x1, y1, x2, y2 = box_ydown
    return [x1, H - y2, x2, H - y1]


def load_cases(split: str, limit: int = 0) -> list[str]:
    if split == "all":
        ids = sorted(d.name for d in Path(OFFICIAL_TRAIN_ROOT).iterdir() if d.is_dir())
    elif split == "golden":
        ids = sorted(d.name.split("-", 1)[0] for d in Path(OFFICIAL_GT_ROOT).iterdir()
                     if d.is_dir())
    else:
        sp = json.loads((Path(__file__).resolve().parents[1] / "data" /
                         "train_val_split.json").read_text(encoding="utf-8"))
        ids = list(sp[split])
    return ids[:limit] if limit else ids


def gt_of_case(case_id: str, source: str = "official-train"):
    """→ `(img_ydown, boxes_ydown, pins_ydown, types)`，全部 y-down。"""
    import numpy as np
    from pcbmodel.data.loader import bbox_flip_down, load_sample
    s = load_sample(case_id, source=source, with_mask=False)
    boxes, types = [], []
    for b in s.comp_boxes:
        x1, y1, x2, y2 = bbox_flip_down(b.bbox_up, s.H)
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        types.append(b.type)
    pins: dict[str, dict[str, tuple[float, float]]] = {}
    for p in s.pin_items:
        y = float(s.H) - p.point_up[1]
        pins.setdefault(p.desig, {})[p.key] = (p.point_up[0], y)
    return s.image_orig, boxes, pins, types, s.comp_boxes, s.H


def predict_case(img, model, plan: TilePlan, *, conf: float, iou: float,
                 device: str = "0") -> list[dict]:
    """切块推理 → 合并 → 全图 y-down 框。"""
    H, W = img.shape[:2]
    tiles = plan_tiles(W, H, plan)
    crops = [img[y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles]
    dets: list[dict] = []
    for rect, crop in zip(tiles, crops):
        res = model.predict(crop, imgsz=plan.tile, conf=conf, iou=iou,
                            device=device, verbose=False)
        for r in res:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            for b, c, s in zip(xyxy, cls, sc):
                norm = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                x0, y0, x1, y1 = rect
                tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
                nb = [(norm[0] + norm[2]) / 2 / tw, (norm[1] + norm[3]) / 2 / th,
                      (norm[2] - norm[0]) / tw, (norm[3] - norm[1]) / th]
                dets.append({"box": to_full_box(nb, rect), "score": float(s),
                             "cls": int(c)})
    return merge_across_tiles(dets)


def match_boxes(preds, gts, iou_min: float = 0.3):
    """贪心按 IoU 配对（pred 高分优先），返回 `[(pred_idx, gt_idx, iou)]`。"""
    from pcbmodel.detect.tiles import iou as _iou
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used: set[int] = set()
    out = []
    for pi in order:
        best, bj = 0.0, None
        for gi in range(len(gts)):
            if gi in used:
                continue
            v = _iou(preds[pi]["box"], gts[gi])
            if v > best:
                best, bj = v, gi
        if bj is not None and best >= iou_min:
            used.add(bj)
            out.append((pi, bj, best))
    return out


def edge_errors(pairs, preds, gts) -> dict:
    ex, ey = [], []
    for pi, gi, _ in pairs:
        p, g = preds[pi]["box"], gts[gi]
        ex += [abs(p[0] - g[0]), abs(p[2] - g[2])]
        ey += [abs(p[1] - g[1]), abs(p[3] - g[3])]
    allv = ex + ey
    if not allv:
        return {"n": 0}
    q = lambda v, p: st.quantiles(v, n=100)[p - 1] if len(v) >= 100 else max(v)
    return {"n_edges": len(allv), "median": st.median(allv),
            "mean": st.mean(allv), "p90": q(allv, 90), "p99": q(allv, 99),
            "median_x": st.median(ex) if ex else 0.0,
            "median_y": st.median(ey) if ey else 0.0}


def pin_error(pairs, preds, gts, gt_boxes, pins, types, desigs, H: float,
              per_type_acc: dict | None = None) -> dict:
    """用**预测框**推导引脚，与 GT 引脚比 —— 这是 A5 存亡的直接答案。

    ⚠⚠ **这里是本仓库最容易踩的跨模块帧陷阱，务必看清**：
      - 本模块与 `tiles.py` 一律 **y-down**（图像原生）
      - 但 `pin_geom.derive_2pin()` **只接受 y-up**（项目 canonical 帧）
    直接把 y-down 框喂进 `derive_2pin` **不会报错**，会给出看似合理的两个点，
    但竖直元件的 `pin_1`/`pin_2` **正好互换** —— 实测命中率 **32.7%→89.6%** 的差距全在这里。
    （水平元件的推导走 x 轴，与帧无关，所以"只有一部分错"，更容易被误判成模型不行。）
    → 正确做法：**推导前转 y-up，推导后再转回 y-down**（下面 `_to_up_box` / 点回转）。
    """
    hit = tot = 0
    per_type: dict[str, list[int]] = {}
    for pi, gi, _ in pairs:
        ty = types[gi]
        if ty not in DERIVABLE_2PIN:
            continue
        got = derive_2pin(_to_up_box(preds[pi]["box"], H), ty)     # ← 必须转 y-up
        if got is None:
            continue
        gp = pins.get(desigs[gi]) or {}
        for k, pt_up in zip(("pin_1", "pin_2"), got):
            g = gp.get(k)
            if g is None:
                continue
            tot += 1
            pt = (pt_up[0], H - pt_up[1])                          # ← 转回 y-down
            ok = abs(g[0] - pt[0]) <= POINT_TOL and abs(g[1] - pt[1]) <= POINT_TOL
            hit += int(ok)
            per_type.setdefault(ty, [0, 0])[0] += int(ok)
            per_type[ty][1] += 1
            if per_type_acc is not None:
                acc = per_type_acc.setdefault(ty, [0, 0])
                acc[0] += int(ok)
                acc[1] += 1
    return {"n_points": tot, "hit": hit,
            "hit_rate": (hit / tot) if tot else 0.0,
            "per_type": {k: round(v[0] / v[1], 3) for k, v in sorted(per_type.items())}}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="eval_box", description="框中位误差 + 引脚命中率评估")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--source", default=None, help="默认按 split 推断 official-gt/train")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    w = Path(a.weights)
    if not w.is_file():
        print(f"[eval_box] 找不到权重 {w}")
        return 2
    model = YOLO(str(w))
    plan = TilePlan(a.tile, a.stride)
    ids = ([x.strip() for x in a.cases.split(",") if x.strip()] if a.cases
           else load_cases(a.split, a.limit))
    source = a.source or ("official-gt" if a.split == "golden" else "official-train")

    all_pairs, all_preds, all_gts = [], [], []
    pin_hit = pin_tot = 0
    per_type: dict[str, list[int]] = {}
    per_case = []
    for cid in ids:
        try:
            img, gboxes, gpins, types, boxes, H = gt_of_case(cid, source)
            preds = predict_case(img, model, plan, conf=a.conf, iou=a.iou,
                                 device=a.device)
        except Exception as e:                                  # noqa: BLE001
            print(f"   ⚠ {cid}: {type(e).__name__}: {e}")
            continue
        pr = match_boxes(preds, gboxes)
        desigs = [b.desig for b in boxes]
        pe = pin_error(pr, preds, gboxes, gboxes, gpins, types, desigs, H,
                       per_type_acc=per_type)
        pin_hit += pe["hit"]
        pin_tot += pe["n_points"]
        st_ = edge_errors(pr, preds, gboxes)
        per_case.append({"case": cid, "n_pred": len(preds), "n_gt": len(gboxes),
                         "n_match": len(pr), "edge_median": st_.get("median")})
        all_pairs.append(pr)

    med = [c["edge_median"] for c in per_case if c["edge_median"] is not None]
    out = {
        "weights": str(w), "split": a.split, "n_cases": len(per_case),
        "n_pred_total": sum(c["n_pred"] for c in per_case),
        "n_gt_total": sum(c["n_gt"] for c in per_case),
        "edge_median_median": st.median(med) if med else None,
        "edge_median_p90": (st.quantiles(med, n=10)[8] if len(med) >= 10
                            else (max(med) if med else None)),
        "pin_points": pin_tot, "pin_hit": pin_hit,
        "pin_hit_rate": round(pin_hit / pin_tot, 4) if pin_tot else 0.0,
        "pin_per_type": {k: round(v[0] / v[1], 3) for k, v in sorted(per_type.items())
                         if v[1]},
        "pin_per_type_n": {k: v[1] for k, v in sorted(per_type.items()) if v[1]},
        "per_case": per_case,
    }
    outp = Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "box_error.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    print(f"[eval_box] weights={w.name} split={a.split} cases={len(per_case)}")
    print(f"  预测框 {out['n_pred_total']} vs GT 框 {out['n_gt_total']}")
    if med:
        print(f"  **逐边误差中位（按例取中位再取中位）= {out['edge_median_median']:.2f}px**")
        print(f"  （按例 p90 = {out['edge_median_p90']:.2f}px）")
    print(f"  引脚点：{pin_hit}/{pin_tot} = **{out['pin_hit_rate'] * 100:.1f}%** 命中（≤5px）")
    print()
    print("  ⚠ 判读（A5 jitter 曲线，GT 框基准）：")
    print("     框误差 ≤2px → pinF1≈0.45 ｜ ~5px → 0.36 ｜ ~10px → 0.10")
    if med:
        m = out["edge_median_median"]
        verdict = ("A5 可用" if m <= 5 else
                   "A5 降级为『只出 key』" if m <= 10 else "A5 基本作废")
        print(f"     本次中位 {m:.2f}px → **{verdict}**")
    print(f"  -> {outp / 'box_error.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
