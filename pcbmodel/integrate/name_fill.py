# -*- coding: utf-8 -*-
"""【隔离候选 2026-10-09】IC 型号补填：给**空 Name 的 box** 补一个型号串。

方案来源 = Codex `prep/work/codex_ic_name_20261009/`（**库确认 + OCR 原文输出**，无模糊、
无库外形态猜测）。本轮把它接进**真实管线**（默认 off），并与位号候选做四臂联合对照。

## 位置：**引脚与网络生成之后**

在 `runner.run_case` 里，`numbering.apply()` 返回的**最终 JSON** 上补写
`components[key]["Name"]` —— 此时 key / pins / nets / bbox / type / value 都已定稿，
本模块**只碰 Name 一个字段**，结构上不可能改变引脚链或拓扑。
（"先让型号进 `DraftComp.name` 再跑引脚链"是另一个实验臂，会改 `box_pins` 的查表输入，
本模块**不做**。）

## 三条硬约定（审核已定）

1. **只补空**：只处理 `ctype == "box"` 且草案 `name` 为空的器件；已有 name 一律不动。
2. **只认库**：候选串必须过 `box_pins.match_model(..., fuzzy1=False)`
   （官方库**精确**命中 + 三道形态闸）⇒ 无模糊纠错、无库外型号形态猜测。
3. **写 OCR 原文**：落进 Name 的是 `OcrText.content` 本身，**不是**库的归一键 ——
   官方是字符串严格匹配，库键 `TYPE-C 16PIN` 与原文 `TYPE-C16PIN` 不是一回事
   （Codex `lib_canonical` 臂因此少对 2 个）。位号形态（`U1`/`D3`/`R12`）被
   `box_pins._JUNK_KEY` 挡在库外 ⇒ **key 不会被抄进 Name**。

⚠ **不改 `box_pins.py` 一个字节**：几何（`face_of`）、距离上限（`MODEL_R`）、形态闸
（`match_model`）全部**复用**（box 引脚链归 Codex）。
"""
from __future__ import annotations

import os

from pcbmodel.integrate import box_pins as _bp

NAME_FILL_ENV = "PCBM_NAME_FILL"

#: 最近一次 `run_case` 的补填记录（**仅开关打开时**被写入；交付入口据此落审核文件）。
#: `None` = **本例没执行到补填**（管线早期就失败）—— 交付入口每例开跑前先重置成 `None`，
#: 失败例**不会**沿用上一例的记录；`[]` = 执行了但零补填（例如图上没有可确认的型号）。
LAST_FILL: list[dict] | None = None

_TRUE = {"1", "true", "yes", "on"}


def name_fill_on() -> bool:
    """开关语义与位号候选**完全一致**：未设 / `0` / 其它 = 关；`1/true/yes/on`（不分大小写）= 开。"""
    return os.environ.get(NAME_FILL_ENV, "").strip().lower() in _TRUE


def _center(t) -> tuple[float, float]:
    """`OcrText.center()` 或 `(content, bbox)` 元组都支持（与 `box_pins._pairs` 同宽容度）。"""
    if hasattr(t, "center"):
        return tuple(float(v) for v in t.center())
    b = t[1]
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _content(t):
    c = getattr(t, "content", None)
    if c is None and not hasattr(t, "content"):
        c = t[0]
    return None if c is None or not str(c).strip() else str(c)


def _bbox(t):
    b = getattr(t, "bbox", None)
    if b is None and not hasattr(t, "bbox"):
        b = t[1]
    return None if b is None else [float(v) for v in b]


def fill_names(comps, texts, *, keys: dict | None = None) -> dict[int, dict]:
    """给**空 name 的 box** 各配一个型号串。返回 `{comp_idx: 记录}`。

    **纯函数**：不改 `comps`、不改 `texts`、不写 `draft` —— 调用方决定怎么用返回值。

    规则（与 Codex `lib_raw` 臂逐条一致）：候选 = 过库闸的文本；距离 = 文本中心到元件框
    的**有符号距离绝对值**（`face_of`，框内为负也算距离）；`<= MODEL_R` 才入候选；
    按 `(距离, 元件序号, 文本序号)` 升序做**全局 1:1** 贪心（一条文本只服务一个元件，
    一个元件只拿一条文本）⇒ 抢不到就**弃权**，不硬塞邻居的型号。
    """
    ids = [i for i, c in enumerate(comps)
           if getattr(c, "ctype", None) == "box" and not getattr(c, "name", None)]
    if not ids or not texts:
        return {}
    if keys is None:
        keys = _bp.lib_keys(_bp.load_db(True)[0])
    cands: list[tuple[float, int, int, str]] = []
    for ti, t in enumerate(texts):
        content = _content(t)
        if content is None:
            continue
        model = _bp.match_model(content, keys, fuzzy1=False)
        if model is None:
            continue                              # 库外 / 形态不过闸 ⇒ 绝不猜
        tc = _center(t)
        for i in ids:
            bb = getattr(comps[i], "bbox", None)
            if bb is None:
                continue
            d = abs(_bp.face_of(tc, [float(v) for v in bb])[1])
            if d <= _bp.MODEL_R:
                cands.append((d, i, ti, model))
    cands.sort(key=lambda x: x[:3])
    used_i: set[int] = set()
    used_t: set[int] = set()
    out: dict[int, dict] = {}
    for d, i, ti, model in cands:
        if i in used_i or ti in used_t:
            continue
        used_i.add(i)
        used_t.add(ti)
        t = texts[ti]
        out[i] = {"text": _content(t), "model": model,
                  "cls": getattr(t, "cls", None), "ti": ti,
                  "distance": round(float(d), 2),
                  "text_bbox": _bbox(t),
                  "comp_bbox": [float(v) for v in comps[i].bbox]}
    return out


def apply_to_result(result: dict, draft, policy=None) -> list[dict]:
    """把补填写进**最终 JSON**（`numbering.apply` 的返回值）。返回逐条记录。

    ⚠ 只写 `components[key]["Name"]`，**不回流进 `draft`** —— 因此 pins / nets / key /
    bbox / type / value 一个都不可能变（要变也变不了：它们已经生成完了）。
    """
    global LAST_FILL
    from pcbmodel.integrate import numbering as N
    texts = getattr(draft, "box_texts", None) or ()
    found = fill_names(draft.comps, texts)
    desigs = N.assign_designators(draft, policy or N.POLICIES["default"])
    comps_out = result.get("components", {})
    rows: list[dict] = []
    for i, rec in sorted(found.items()):
        if i >= len(desigs) or desigs[i] not in comps_out:
            continue
        comps_out[desigs[i]]["Name"] = rec["text"]
        rows.append({"key": desigs[i], **rec})
    LAST_FILL = rows
    return rows


# --------------------------------------------------------------------------- #
# 自检（不依赖数据集/torch）
# --------------------------------------------------------------------------- #
def selftest() -> int:
    from types import SimpleNamespace as S

    def txt(content, x, y, cls="text_other"):
        bb = [x - 5.0, y - 5.0, x + 5.0, y + 5.0]
        return S(content=content, cls=cls, bbox=bb,
                 center=lambda: ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0))

    def box(bb, name=None):
        return S(ctype="box", name=name, bbox=bb, desig_text=None, pins=[])

    keys = _bp.lib_keys(_bp.load_db(True)[0])
    K = next(k for k in keys if _bp.model_like(k, keys))
    raw = K[:2] + " " + K[2:]                    # 库里是 K，原文多一个空格 ⇒ 归一后同键

    got = fill_names([box([0, 0, 10, 10]), box([20, 0, 30, 10], name="KEEP")],
                     [txt(raw, 15, 5)], keys=keys)
    assert list(got) == [0], got                 # 已有 name 的不动
    assert got[0]["text"] == raw, got            # ⚠ 写**原文**，不是库键 K
    assert got[0]["model"] == K, got
    assert fill_names([box([0, 0, 10, 10])], [txt("U1", 5, 5)], keys=keys) == {}
    assert fill_names([box([0, 0, 10, 10])], [txt(raw, 5000, 5)], keys=keys) == {}
    # 型号形态但不是库内键 ⇒ 弃权（**不做库外形态猜测**，`lib_raw_unknown` 臂已停线）
    assert fill_names([box([0, 0, 10, 10])], [txt("ZZQQ9999XY", 5, 5)], keys=keys) == {}
    twin = fill_names([box([0, 0, 10, 10]), box([20, 0, 30, 10])], [txt(raw, 15, 5)], keys=keys)
    assert len(twin) == 1, twin                  # 1:1：一条文本不服务两个元件
    # 结构不变量：只有 Name 会变
    res = {"components": {"U1": {"Name": "U1", "type": "box", "value": None,
                                 "bbox": [0.0, 0.0, 10.0, 10.0]}},
           "pins": {"U1": {"pin_1": {"pinname": "VCC", "point": {"x": 1.0, "y": 2.0}}}},
           "nets": {"net_1": {"hyperGraph": "(U1.1)", "edges": {}}}}
    import copy
    before = copy.deepcopy(res)
    d = S(comps=[box([0, 0, 10, 10])], box_texts=[txt(K, 5, 5)])
    rows = apply_to_result(res, d)
    assert [r["key"] for r in rows] == ["U1"] and res["components"]["U1"]["Name"] == K, rows
    for k in ("pins", "nets"):
        assert res[k] == before[k], k
    assert {k: v for k, v in res["components"]["U1"].items() if k != "Name"} == \
           {k: v for k, v in before["components"]["U1"].items() if k != "Name"}
    print("name_fill selftest OK（只补空/只认库/写原文/位号不当型号/超距弃权/1:1/只改 Name）")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
