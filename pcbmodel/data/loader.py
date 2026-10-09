# -*- coding: utf-8 -*-
"""P1 阶段 0.3：每例 sample 装配（官方标签为准，四类监督一次给齐）。

数据源（决策 12/15，2026-09-10 拍板）：
  * **标签 = 官方 target**（`赛题六公开数据集/200_train_cases`，或 `10_GTcase`）：
    `components`(Name 大写 / 官方 type 词表 / bbox) + `pins` + `nets`。
    官方坐标**已是 y-up**，与 canonical 同系 → 元件框/引脚点**不做任何翻转**。
  * **文本监督仍回源 CVJsonStd**（`Dataset300/<id>/`，**300 例全有**——含 0201–0300，
    它们只是文件名是 hash、图是 jpg/png 混合；2026-09-10 实测订正）：官方 target 无
    文本框，OCR 检测/识别监督只能用源 `texts.location + details + content`。
  * **U-Net 掩码 = 官方 `nets.edges`**（决策 15，取代此前"源 edges 全量"）。

坐标系（决策 2）：sample 里 `image_up` 与所有几何（bbox_up/point_up/wire_mask_up/
gt_nets）同为 canonical y-up；唯一保留 `image_orig`(y-down) 供 OCR 识别直立裁剪
（`crop_text_orig` 现算）。本模块与 `masks.py` 是仅有的两处涉及 y-down 的地方。

OCR 监督口径（决策 3）：`details` 取 {designator,value,pinNumber,pinName,netlabel}
+ `searchName`→`name`；`details=''`（约 41%）**默认排除**；content 噪声（∅/null/空）剔除。

`details=''` 的可选透出（2026-09-17）：`load_sample(..., include_unclassified=True)`
把**空 details**的框以 `detail=TEXT_OTHER` 留下（**默认关 ⇒ 上面那行口径逐字不变**）。
检测侧（`ocr/dataset_text.py`）本来就把它们收成第 5 类，**从来没收不到**；丢的是**识别**
那一环 —— `dataset_recog` 迭代的就是 `_text_boxes`。规范名未知（非空但不认识）的
**仍不放行**，与 `probe_text_supervision` 数出 210 条时口径一致。见 MODEL_HANDOFF §九。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.caseio import (IMG_EXTS, CaseError, discover_case_files,
                            discover_official, load_cvjsonstd, load_image)
from pcbeval.config import EvalConfig, OFFICIAL_GT_ROOT, OFFICIAL_TRAIN_ROOT
from pcbeval.gtbuilder import is_noise
from pcbmodel.data import masks

# 官方数据源 -> 根目录（kind 由键决定）
OFFICIAL_SOURCES: dict[str, tuple[Path, str]] = {
    "official-train": (OFFICIAL_TRAIN_ROOT, "train"),
    "official-gt": (OFFICIAL_GT_ROOT, "gt"),
}

OCR_CLASSES: tuple[str, ...] = ("designator", "value", "pinNumber", "pinName",
                                "netlabel", "name")
# `details=''` 的兜底类名（检测侧 `dataset_text` 的第 5 类）。**故意不进 OCR_CLASSES** ——
# 它是"未分类"，不是可训练标注；只有 `include_unclassified=True` 才出现。
TEXT_OTHER = "text_other"
_DETAIL_MAP = {
    "designator": "designator", "value": "value",
    "pinnumber": "pinNumber", "pinname": "pinName",
    "netlabel": "netlabel",
    "searchname": "name", "name": "name",
}


def normalize_detail(details: str | None) -> str | None:
    """源 texts.details -> OCR 类规范名；''/未知 返回 None（决策 3 排除）。"""
    if not details:
        return None
    return _DETAIL_MAP.get(details.strip().lower())


def bbox_flip_up(b: dict, H: float) -> list[float]:
    """源 y-down box {x1,y1,x2,y2} -> y-up [x1, H-y2, x2, H-y1]（仅 CVJsonStd 文本用）。"""
    return [float(b["x1"]), H - float(b["y2"]), float(b["x2"]), H - float(b["y1"])]


def bbox_flip_down(bbox_up: list[float], H: float) -> tuple[int, int, int, int]:
    """y-up bbox [x1,y1,x2,y2] -> y-down 整数 (x1, y_top, x2, y_bot)。"""
    x1, y1u, x2, y2u = (float(v) for v in bbox_up)
    return int(round(x1)), int(round(H - y2u)), int(round(x2)), int(round(H - y1u))


@dataclass
class Box:
    desig: str
    type: str
    name: str | None
    value: str | None
    bbox_up: list[float]


@dataclass
class Pin:
    desig: str
    key: str            # pin_<n>。命名元件=图上物理脚号（可学）；∅ 元件=计数器（不可复现）。见 MODEL_HANDOFF §零
    pinname: str | None
    point_up: list[float]


@dataclass
class TextBox:
    tid: str
    detail: str                 # OCR_CLASSES 之一；`include_unclassified` 时可为 TEXT_OTHER
    content: str | None
    bbox_up: list[float]


@dataclass
class Sample:
    case_id: str
    source: str
    W: int
    H: int
    image_up: np.ndarray                 # [H,W,3] uint8 canonical y-up
    image_orig: np.ndarray               # [H,W,3] uint8 原图 y-down（OCR 裁图）
    comp_boxes: list[Box] = field(default_factory=list)
    pin_items: list[Pin] = field(default_factory=list)
    text_boxes: list[TextBox] = field(default_factory=list)
    wire_mask_up: np.ndarray | None = None
    gt_nets: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)   # 官方 target 原样（便于后续模块取用）


def _comp_boxes(target: dict) -> list[Box]:
    out = []
    for desig, c in target.get("components", {}).items():
        out.append(Box(desig=desig, type=c.get("type", "other"),
                       name=c.get("Name", c.get("name")),
                       value=c.get("value"),
                       bbox_up=[float(v) for v in c["bbox"]]))
    return out


def _pins(target: dict) -> list[Pin]:
    out = []
    for desig, pd in target.get("pins", {}).items():
        for key, p in pd.items():
            pt = p.get("point") or {}
            if "x" not in pt or "y" not in pt:
                continue
            out.append(Pin(desig=desig, key=key, pinname=p.get("pinname"),
                           point_up=[float(pt["x"]), float(pt["y"])]))
    return out


def _text_boxes(cv: dict, H: float,
                include_unclassified: bool = False) -> list[TextBox]:
    out = []
    for tid, t in cv.get("texts", {}).items():
        raw = t.get("details")
        detail = normalize_detail(raw)
        if detail is None:
            # ⚠ 只放行**真空**的 `details`；非空但不认识的一律照旧丢（口径同
            #   `probe_text_supervision`，那 210 条就是这么数出来的）。
            if not (include_unclassified and not str(raw or "").strip()):
                continue
            detail = TEXT_OTHER
        content = t.get("content")
        if is_noise(content):
            continue
        loc = t.get("location")
        if not loc or "x1" not in loc:
            continue
        if float(loc["x2"]) - float(loc["x1"]) <= 0 or \
           float(loc["y2"]) - float(loc["y1"]) <= 0:
            continue
        out.append(TextBox(tid=tid, detail=detail, content=content,
                           bbox_up=bbox_flip_up(loc, H)))
    return out


# 交付模式（`source="image"`）的图片目录 —— 官方 /input 只挂图片、不挂 GT。
# 由 `integrate/deliver.py` 在跑批前设一次。见 MODEL_HANDOFF §五 4.8。
IMAGE_DIR: Path | None = None


def _image_case(case_id: str) -> Path:
    """交付模式找图：case_id 是图片文件名去扩展名（**可含空格**，如 `0025-Altium Designer`）。"""
    if IMAGE_DIR is None:
        raise CaseError('source="image" 需要先设 loader.IMAGE_DIR（见 integrate/deliver.py）')
    root = Path(IMAGE_DIR)
    p = root / f"{case_id}.png"                       # 官方约定就是 .png，先走这条
    if p.is_file():
        return p
    # ponytail: O(n) 扫描。官方是 30~50 例量级，无所谓；上千例再建一次索引。
    got = [q for q in sorted(root.iterdir())
           if q.stem == case_id and q.suffix.lower() in IMG_EXTS]
    if not got:
        raise CaseError(f"交付图片未找到: {root}/{case_id}.*")
    return got[0]


def load_image_sample(case_id: str) -> Sample:
    """**无 GT** 的裸图 sample —— 官方交付口径（`/input` 里只有 PNG）。

    只填图像派生字段（`image_up` / `image_orig` / `W` / `H`），GT 字段（comp_boxes /
    pin_items / text_boxes / wire_mask_up / gt_nets / target）**一律留空**。
    真端到端的三条路径（`detect`、`ocr-real(boxes=detect)`、`--nets wire`）本来就不读 GT
    —— 所以下游一行都不用改，这正是接进官方入口的全部代价。
    """
    pil = load_image(_image_case(case_id)).convert("RGB")
    W, H = pil.size
    image_orig = np.asarray(pil, dtype=np.uint8)
    return Sample(case_id=case_id, source="image", W=W, H=H,
                  image_up=np.flipud(image_orig), image_orig=image_orig)


def load_sample(case_id: str, *, source: str = "official-train",
                with_mask: bool = True, official_root: Path | None = None,
                cfg: EvalConfig | None = None,
                include_unclassified: bool = False) -> Sample:
    """装配一例 sample。source ∈ {official-train, official-gt, image}。

    文本监督回源 Dataset300 的 CVJsonStd（按 case_id 发现，hash 命名也能找到）。
    Dataset300 全部 300 例都有 CVJsonStd；走 official-train/official-gt 时 case 本身
    限定在 200_train/10_GT 内，故这里实际不会缺。仍保留 try/except 容错。

    `include_unclassified=True`：额外把 `details=''` 的框以 `detail=TEXT_OTHER` 收进
    `text_boxes`（默认 False ⇒ 与历史行为逐字一致）。**谁放行谁负责判** —— 这些框没有
    标注，下游必须自己决定要不要（`dataset_recog._label_for` 只收 content 逐字等于某
    GT 位号的）。交付路径**不要开**。
    """
    cfg = cfg or EvalConfig()
    if source == "image":                       # 交付：无 GT，只有图
        return load_image_sample(case_id)
    if official_root is not None:
        root, kind = Path(official_root), "auto"
    elif source in OFFICIAL_SOURCES:
        root, kind = OFFICIAL_SOURCES[source]
    else:
        raise ValueError(f"未知 source: {source}")
    oc = discover_official(case_id, root, kind=kind)
    with open(oc.target, encoding="utf-8") as f:
        target = json.load(f)

    if oc.image is None:
        raise CaseError(f"官方 case {case_id} 缺图片")
    pil = load_image(oc.image).convert("RGB")
    W, H = pil.size
    image_orig = np.asarray(pil, dtype=np.uint8)
    image_up = np.flipud(image_orig)

    text_boxes: list[TextBox] = []
    try:
        cv = load_cvjsonstd(discover_case_files(case_id, cfg).cvjson)
        text_boxes = _text_boxes(cv, H, include_unclassified=include_unclassified)
    except CaseError:
        pass  # 无 CVJsonStd（官方 0201+）→ 无文本监督

    return Sample(
        case_id=case_id, source=source, W=W, H=H,
        image_up=image_up, image_orig=image_orig,
        comp_boxes=_comp_boxes(target),
        pin_items=_pins(target),
        text_boxes=text_boxes,
        wire_mask_up=masks.rasterize_target_wire_mask(target, W, H) if with_mask else None,
        gt_nets=target.get("nets", {}),
        target=target,
    )


def crop_text_orig(sample: Sample, i: int, pad: int = 4) -> np.ndarray:
    """按 sample.text_boxes[i].bbox_up 从 y-down 原图现算直立裁剪（含 pad）。

    仅这里做 y-up -> y-down 的框换算；返回 [cropH, cropW, 3] uint8。越界自动裁剪。
    """
    x1, y_top, x2, y_bot = bbox_flip_down(sample.text_boxes[i].bbox_up, sample.H)
    x1 = max(0, x1 - pad); y_top = max(0, y_top - pad)
    x2 = min(sample.W, x2 + pad); y_bot = min(sample.H, y_bot + pad)
    return sample.image_orig[y_top:y_bot, x1:x2]


def main(argv: list[str] | None = None) -> int:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="loader", description="单例 sample 冒烟")
    ap.add_argument("case_id")
    ap.add_argument("--source", default="official-train",
                    choices=list(OFFICIAL_SOURCES))
    ap.add_argument("--no-mask", action="store_true")
    a = ap.parse_args(argv)
    s = load_sample(a.case_id, source=a.source, with_mask=not a.no_mask)
    mask = int(s.wire_mask_up.sum()) if s.wire_mask_up is not None else None
    print(f"[loader] {s.case_id}({s.source}) HxW={s.H}x{s.W} "
          f"comps={len(s.comp_boxes)} pins={len(s.pin_items)} texts={len(s.text_boxes)} "
          f"mask.sum={mask} nets={len(s.gt_nets)} "
          f"img_up==flipud(img_orig):{bool(np.array_equal(s.image_up, np.flipud(s.image_orig)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
