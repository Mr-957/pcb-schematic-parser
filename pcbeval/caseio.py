# -*- coding: utf-8 -*-
"""case 文件发现与读取（Dataset300 只读）。

每例目录 Dataset300/<id>/ 下通常有：
  <id>-<tool>_CVJsonStd.json  内部 golden（components/pins/texts/edges）
  <id>-<tool>_spice.net       官方流水线产出的 SPICE（含权威 net 分组注释）
  <id>-<tool>.eds             EDIF 网表（本 P0 不读）
  <id>.png|jpg                原理图
文件名前缀不固定（tool 名/哈希），一律按后缀发现。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import EvalConfig

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class CaseError(FileNotFoundError):
    pass


@dataclass
class CaseFiles:
    case_id: str
    cvjson: Path | None = None
    net: Path | None = None
    image: Path | None = None
    extra: list = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [k for k in ("cvjson", "net", "image") if getattr(self, k) is None]


def discover_case_files(case_id: str, cfg: EvalConfig | None = None) -> CaseFiles:
    cfg = cfg or EvalConfig()
    d = cfg.dataset_root / case_id
    if not d.is_dir():
        raise CaseError(f"case 目录不存在: {d}")
    files = sorted(p for p in d.iterdir() if p.is_file())
    cf = CaseFiles(case_id=case_id)
    for p in files:
        name = p.name.lower()
        if name.endswith("_cvjsonstd.json"):
            cf.cvjson = p
        elif name.endswith("_spice.net") or name.endswith(".net"):
            cf.net = p
        elif p.suffix.lower() in IMG_EXTS and cf.image is None:
            cf.image = p
        else:
            cf.extra.append(p)
    if cf.cvjson is None:
        raise CaseError(f"case {case_id} 缺少 *_CVJsonStd.json")
    return cf


@dataclass
class OfficialCase:
    """官方公开数据集（2026-09 到手）的一例：PNG + 官方 target JSON。

    两种布局：
      train: <root>/<case_id>/<case_id>-<tool>_target_named.json
      gt   : <root>/<case_id>-<tool>/<case_id>-<tool>_target.json
    官方 target 无 CVJsonStd；如需 `texts`/导线折线监督须回源 Dataset300。
    """
    case_id: str
    kind: str
    root: Path
    dir: Path
    image: Path | None = None
    target: Path | None = None


def _pick_target(d: Path) -> Path | None:
    """目录内挑 target JSON：优先 *_target.json，其次 *_target_named.json。"""
    cands = sorted(d.glob("*_target*.json"))
    if not cands:
        return None
    plain = [p for p in cands if p.name.endswith("_target.json")]
    named = [p for p in cands if p.name.endswith("_target_named.json")]
    return (plain or named)[0]


def discover_official(case_id: str, root: Path, kind: str = "auto") -> OfficialCase:
    """官方数据集 case 发现。kind ∈ {train, gt, auto}；都找不到 raise CaseError。

    auto 先试 train 布局（<id>/），再试 gt 布局（<id>-<tool>/）。
    """
    root = Path(root)
    if not root.is_dir():
        raise CaseError(f"官方数据根不存在: {root}")
    kinds = ["train", "gt"] if kind == "auto" else [kind]
    for k in kinds:
        if k == "train":
            dirs = [root / case_id] if (root / case_id).is_dir() else []
        else:
            dirs = sorted(d for d in root.iterdir()
                          if d.is_dir() and d.name.startswith(f"{case_id}-"))
        for d in dirs:
            target = _pick_target(d)
            if target is None:
                continue
            img = next((p for p in sorted(d.iterdir())
                        if p.suffix.lower() in IMG_EXTS), None)
            return OfficialCase(case_id=case_id, kind=k, root=root, dir=d,
                                image=img, target=target)
    raise CaseError(f"官方数据未找到 case {case_id}（root={root}, kind={kind}）")


def load_cvjsonstd(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_target(path: Path) -> dict:
    """读一份 target 格式（components/pins/nets）JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path: Path) -> Image.Image:
    return Image.open(path)


def canvas_vs_image(cv: dict, img: Image.Image):
    """CVJsonStd.canvas vs 图片实际尺寸。返回 (canvas, img_size, ok, note)。"""
    cw, ch = cv.get("canvas", {}).get("width"), cv.get("canvas", {}).get("height")
    iw, ih = img.size
    ok = (cw, ch) == (iw, ih)
    note = "" if ok else f"canvas=({cw},{ch}) != image=({iw},{ih})，绘图需缩放"
    return (cw, ch), img.size, ok, note


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法: python -m pcbeval.caseio <case_id> [case_id ...]")
        return 2
    cfg = EvalConfig()
    bad = 0
    for cid in argv:
        try:
            cf = discover_case_files(cid, cfg)
            cv = load_cvjsonstd(cf.cvjson)
            print(f"[{cid}] cvjson={cf.cvjson.name}")
            if cf.net:
                print(f"[{cid}] net   ={cf.net.name}")
            if cf.image:
                img = load_image(cf.image)
                canvas, isize, ok, note = canvas_vs_image(cv, img)
                print(f"[{cid}] img   ={cf.image.name} size={isize} canvas={canvas} match={ok} {note}")
            if cf.missing:
                print(f"[{cid}] 缺少: {cf.missing}  (extra={[p.name for p in cf.extra][:6]})")
        except CaseError as e:
            print(f"[{cid}] ERROR: {e}")
            bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main())
