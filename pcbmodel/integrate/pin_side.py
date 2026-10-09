# -*- coding: utf-8 -*-
"""**pin_1 侧向预测**（吃图像；`pin_geom` 保持纯几何，本模块是它的"上料层"）。

分层（2026-09-22 定）：`pin_geom.derive_2pin(bbox, ctype, *, pin1_side=None)` **只吃框**；
本模块吃**图像**产出 `"low"|"high"|None`，由 `runner.ensure_pins`（手上同时有框与原图）
经**已有的** `pin1_side=` 注入点传入。

`source`：`off`（默认，永远 None = 历史行为逐位不变）/ `geo`（极性标记）/ `model`（图像分类器）
/ `parallel`（分类器为主 + 几何交叉校验：几何说不一致 ⇒ None）。

⚠⚠ **常数表 100% 的类型永不被覆盖**（r/c/l/fuse/crystal_*）⇒ `side_of()` 直接返回 None。
⚠ 方向不许硬编码：v6 实测（d 66.5% / led 87.0% / box 78.0% / fuse 87.5%）四类**都是原向**
   ⇒ "标记在低端"就判 pin_1 在低端；改这里必须附新的一致率表来源。

跑：`python -m pcbmodel.integrate.pin_side --selftest`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NEVER_COVER = frozenset({"r", "c", "l", "fuse", "crystal_2pin", "crystal_3pin",
                         "crystal_4pin", "crystal"})
_FROZEN_FLIP: dict[str, bool] = {}          # 见模块 docstring：四类都是"原向"
GEO_MIN_STRENGTH = 0.6
MODEL_MIN_PROB = 0.65
_MODEL_CACHE: dict = {}


def _geo_side(img, bbox_up, ctype):
    """→ `(side, strength)`；复用 `probe_polarity_marks` 的判据与按类 pad。"""
    from pcbmodel.integrate.probe_polarity_marks import (BRANCH, PAD_BY_TYPE,
                                                         axis_profile, mark_side)
    prof = axis_profile(img, bbox_up, ctype, PAD_BY_TYPE.get(ctype, 0))
    if prof is None:
        return None, 0.0
    dark, axis_body, perp_body = prof
    fn = BRANCH.get(ctype, mark_side)
    r = (fn(dark, GEO_MIN_STRENGTH, perp_body, axis_body) if fn is mark_side
         else fn(dark, GEO_MIN_STRENGTH, perp_body))
    if r is None:
        return None, 0.0
    side, strength, _i = r
    is_low = (side < 0) != bool(_FROZEN_FLIP.get(ctype, False))
    return ("low" if is_low else "high"), float(strength)


def load_model(path=None) -> dict:
    """读 `train_pin1_side --out` 的 npz（进程内缓存）。"""
    p = Path(path) if path else (ROOT / "pcbmodel" / "runs" / "stage0" / "pin1_side.npz")
    if str(p) in _MODEL_CACHE:
        return _MODEL_CACHE[str(p)]
    if not p.is_file():
        raise FileNotFoundError(f"side_source=model 需要权重：{p}（先跑 train_pin1_side --out ...）")
    z = np.load(p, allow_pickle=True)
    m = {"X": z["X"], "y": z["y"], "types": z["types"], "lam": float(z.get("lam", 1e-2))}
    _MODEL_CACHE[str(p)] = m
    return m


def _model_side(img, bbox_up, ctype, model=None):
    """→ `(side, prob)`；按类型取子集建线性核岭（与 `train_pin1_side` 同口径）。"""
    from pcbmodel.integrate.train_pin1_side import _axis, _feat
    m = model if model is not None else load_model()
    types, X, y = m["types"], m["X"], m["y"]
    sel = np.asarray([t == ctype for t in types]) if len(types) else np.zeros(0, bool)
    if int(sel.sum()) < 20:
        return None, 0.0
    Xt, yt = X[sel], y[sel].astype(float)
    if len(set(yt.tolist())) < 2:
        return None, 0.0                       # 单类 ⇒ 无信息
    x = _feat(img, bbox_up, _axis(list(bbox_up), ctype))
    n = len(yt)
    A = np.linalg.solve(Xt @ Xt.T / Xt.shape[1] + m["lam"] * n * np.eye(n), yt)
    score = float((x @ Xt.T / Xt.shape[1]) @ A)
    prior = float(yt.mean())
    prob = float(0.5 + 0.5 * np.tanh(score - prior))
    return ("high" if prob >= 0.5 else "low"), max(prob, 1.0 - prob)


def side_of(img, bbox_up, ctype, *, source="off", model=None):
    """→ `"low"|"high"|None`；`None` = 交回 `pin_geom.PIN1_SIDE` 常数表。"""
    if source == "off" or ctype in NEVER_COVER:
        return None
    if source == "geo":
        return _geo_side(img, bbox_up, ctype)[0]
    if source == "model":
        side, prob = _model_side(img, bbox_up, ctype, model)
        return side if (side is not None and prob >= MODEL_MIN_PROB) else None
    if source == "parallel":
        mside, prob = _model_side(img, bbox_up, ctype, model)
        if mside is None or prob < MODEL_MIN_PROB:
            return None                        # 分类器不敢下结论 ⇒ 回退常数表
        gside, _s = _geo_side(img, bbox_up, ctype)
        return None if (gside is not None and gside != mside) else mside
    raise ValueError(f"未知 source={source!r}（off/geo/model/parallel）")


def selftest() -> int:
    """纯合成图自检：三条硬规则 + 横杠两端都能判 + 空模型安全。"""
    H = W = 40

    def make(bar_row):
        """**空心轮廓**的本体（上下边也是整条横线！）+ 一条横跨本体的细横杠。

        ⚠ 这个 selftest 顺手暴露了一条真问题：**本体自身的上下边与阴极条同形**（都是整条横线）
        ⇒ 覆盖率曲线两端天然有尖峰，会把"哪端是阴极条"搅浑（很可能是 d 卡在 66% 的一部分原因；
        下一步候选判据 = 只在**本体内部中段**找尖峰 + 排除与上下边同宽同位置的行）。
        """
        img = np.full((H, W), 255, np.uint8)
        img[6, 10:30] = 0
        img[37, 10:30] = 0                     # 上/下边（整条横线，与横杠同形）
        img[6:38, 10] = 0
        img[6:38, 29] = 0                      # 左右边
        if bar_row is not None:
            img[bar_row, 10:30] = 0
        return img

    bbox = [10.0, 2.0, 30.0, 34.0]             # y-up：对应 y-down 行 6..38
    assert side_of(make(None), bbox, "d", source="off") is None
    for ty in sorted(NEVER_COVER):             # 硬规则：常数表类型永不覆盖
        assert side_of(make(10), bbox, ty, source="geo") is None, ty
        assert side_of(make(10), bbox, ty, source="model") is None, ty
    # 横杠放在**中段**（离上下边 ≥6px），否则会与"上下边"的尖峰混在一起（见 make 的 docstring）
    assert side_of(make(30), bbox, "d", source="geo") == "low"    # 中下段 ⇒ 低端
    assert side_of(make(12), bbox, "d", source="geo") == "high"   # 中上段 ⇒ 高端
    empty = {"X": np.zeros((0, 4)), "y": np.zeros(0), "types": np.array([]), "lam": 1e-2}
    assert side_of(make(30), bbox, "d", source="model", model=empty) is None
    assert side_of(make(30), bbox, "d", source="parallel", model=empty) is None
    print("[pin_side] selftest OK（off/常数表短路 ｜ geo 两端可判 ｜ 空模型安全返回 None）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pin_side", description="pin_1 侧向预测（吃图像）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
