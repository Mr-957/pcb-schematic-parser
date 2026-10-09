# -*- coding: utf-8 -*-
"""1.3 文本 → 元件/引脚匹配（content 匹配键 + 几何 tiebreak）。

共识（STATUS 待办 3 / MODEL_HANDOFF §五 D10）：**匹配键是 OCR 读到的
content 本身，不是最近邻几何** —— 最近邻会把位号判给隔壁元件（"重贴投毒"）。

规则（统一口径，不依赖 GT；GT 只用于事后评估）：
- designator 类：content 形似位号（`R5`/`U1`/`Q12A`/`C103`…）且距元件 ≤40px → 赋位号
- pinNumber  类：content 形似脚号（`2`/`19`/`3B`…）且距引脚 ≤22px → 赋物理脚号
- pinName    类：距引脚 ≤22px → 赋引脚名
- value      类：距元件 ≤40px → 赋元件值

冲突一律几何 tiebreak（最近者胜）。netlabel/text_other 本阶段不参与（net 构建属阶段 2）。
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field

from pcbmodel.ocr.predict_ocr import OcrText

MAX_DESIG_DIST = 50  # v9: 40→50，c 桶实测 6 个读对文本在 40.5-45.9px 边缘被拒（C20/C13/C15/C10/C12/U14）.0
MAX_PIN_DIST = 22.0
MAX_VALUE_DIST = 40.0

_DESIG_RE = re.compile(
    r"^(?:[A-Za-z]{1,3}\d{1,4}(?:\.\d{1,4})?[A-Za-z]?|[A-Za-z]{1,3}_\d{1,4}[A-Za-z]?)$"
)   # R5, U1, Q12A, C103, U2.3（点位号）, TP_1（下划线位号）
    # 注：`.`/`_` 形态与版本号（V1.0）同构，救回是覆盖优先，风险在评估里看数字
_PIN_RE = re.compile(r"^\d{1,3}[A-Za-z]?$")                   # 2, 19, 3B
# value 形态：数字+**至少 1 个单位字母**，如 2.2uF / 22kΩ / 10Ω / 100nF / 10k / 100K / +5V
# 纯数字（10.0/3499）与 1-2 位短脚号（3B/19）不进 value 通道
_VALUE_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?\s*[A-Za-zΩµμu%‰]+$")
_NETNAME_RE = re.compile(r"^\+?[A-Za-z][A-Za-z0-9_]{1,11}$")  # GND/VCC/VIN/VOUT/+3V3
_SHORT_PIN_RE = re.compile(r"^[+-]?\d{1,2}[A-Za-z]?$")        # 3B/19 短脚号形态

# 2026-09-17 v5：desig 通道 type 过滤（probe_type_prefix.py 实测，val19 GT）——
# 1) **只对"GT key 是位号形"的 type 开放**：gnd/v/net_short/jumper/m3螺丝/buzzer/motor/
#    battary/ant/block/circle_header/dc/esd/opt/other/transfomer 的 GT key 完全不是
#    位号形（desig 形态 0），现在却在抢位号形文本（实测 0199/C18 被 ∅12(gnd) 抢配）。
# 2) **位号首字母必须匹配元件 type**：GT 实测 r→R(158/158)、c→C(114/114)、d→D(31/31)、
#    l→L、bjt/mosfet→Q、switch→S(17/17)、pin/testpoint→T、fuse→F、amp*→U 全部 100%；
#    led→{L,D}、box→{U 占 94.6%}。杀"形似位号的假文本抢占"（0106 电容被 I5p 占、
#    0199 box 被 I9 占、0130 box 被 H1 占、0183 USB2 周围一圈 A/B/E 噪声）。
_DESIG_TYPES = frozenset({
    "r", "c", "d", "box", "led", "l", "switch", "bjt", "mosfet",
    "crystal_2pin", "crystal_3pin", "crystal_4pin",
    "pin", "testpoint", "fuse", "amp", "amp_3pin", "amp_5pin",
})
_TYPE_PREFIX = {
    "r": frozenset("R"), "c": frozenset("C"), "d": frozenset("D"),
    # box 首字母放宽到 GT 实测全集 {U,J,L,O,C,H}（0170 一批连接器/插座被 YOLO 分成 box，
    # GT 位号是 JP1/JP2/LD1/LD2/CH1/CN2/CVY1/H1/H3/H22/o1——全量过滤误杀 -2.9 分）；
    # I/A/B/E 等假文本首字母仍被拒（0106 I5p / 0199 I9 / 0183 USB2 附近 A5/B7/E0）
    "box": frozenset("UJLOCH"), "led": frozenset("LD"),
    "l": frozenset("L"), "switch": frozenset("S"),
    "bjt": frozenset("Q"), "mosfet": frozenset("Q"),
    "crystal_2pin": frozenset("CXY"), "crystal_3pin": frozenset("CXY"),
    "crystal_4pin": frozenset("CXY"),
    "pin": frozenset("T"), "testpoint": frozenset("T"), "fuse": frozenset("F"),
    "amp": frozenset("U"), "amp_3pin": frozenset("U"), "amp_5pin": frozenset("U"),
}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


# ══════════════════════════════════════════════════════════════════════════════
# 【隔离候选 2026-10-09】text_other 明确位号 → designator 校正
#
# 开关：环境变量 `PCBM_DESIG_RECLASS`（`1`/`true`/`yes` 开，**其余一律关**）。
#   **默认关** ⇒ `match()` 的输入与本文件其余部分完全不产生交互，交付逐字节复现
#   现行 40/40（见 `tests/test_desig_reclass.py::TestSwitchOffIsInert` 与
#   `check_fidelity.py`）。
# 打开后**只**把 `cls=="text_other"` 且明确是位号的文本就地改成 `designator`：
#   不改 content / bbox / conf / 列表顺序 / 索引，不新增或删除文本，不使用 GT，
#   不改任何距离（value 仍 40，位号仍 `MAX_DESIG_DIST`，引脚仍 `MAX_PIN_DIST`）。
# 目的：明确位号不再落进 value 的**第二轮宽正则 fallback**（`_NETNAME_RE` 过宽，
#   会把 `C12`/`VIN` 一类串当 value 填给元件）。
# 来源与口径见 `prep/work/claude_designator_candidate_20261009/README.md`。
# ══════════════════════════════════════════════════════════════════════════════

DESIG_RECLASS_ENV = "PCBM_DESIG_RECLASS"
#: 最近一次 `draft_from_ocr` 的提升记录（仅开关打开时被写入；交付入口据此落审核文件）。
#: `None` = **本例没执行到校正**（OCR/检测早期就失败）—— 交付入口每例开跑前先重置成
#: `None`，因此失败例**不会**沿用上一例的记录；`[]` = 执行了但零提升。二者要分清。
LAST_RECLASS: list[dict] | None = None

# 数值形态（与 Codex 实验 dimension()/numeric 判定逐条等价，见 README §口径）
_NUM_UNIT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[pnumk]?F", re.I)
_NUM_OHM = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[kmgmun]?(?:Ω|ohms?|R)", re.I)
_NUM_BARE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[pnumk]", re.I)   # 100n / 4k
_NUM_EMBED = re.compile(r"\d+[RrKkMmNnPpUu]\d+")                       # 4K7 / 4R7 / 4n7


def desig_reclass_on() -> bool:
    """显式开关。交付入口只读环境变量，未设 = 关。"""
    return os.environ.get(DESIG_RECLASS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _desig_norm(s: str) -> str:
    """位号/数值判定统一用的归一化：NFKC + 去空白 + μ→u（大小写留待 `norm`）。"""
    s = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(s)))
    return s.replace("μ", "u").replace("µ", "u")


def _is_numeric_text(s: str) -> bool:
    """`s` 已经是 `_desig_norm` 的形式。数值文本**不**提升为位号。"""
    return bool(_NUM_UNIT.fullmatch(s) or _NUM_OHM.fullmatch(s)
                or _NUM_BARE.fullmatch(s) or _NUM_EMBED.fullmatch(s))


def reclassify_text_other(texts: list, comps: list) -> list[dict]:
    """把 `text_other` 里**明确是位号**的文本就地改成 `cls="designator"`。

    判定（三条全中才提升；无 GT 参与）：
      1. 不是数值形态（`100nF`/`4K7`/`10kΩ` 一类不提升）；
      2. 过生产 `_DESIG_RE`（`norm` 后整串匹配）；
      3. 在**现有** `MAX_DESIG_DIST` 内有器件，其 type ∈ `_DESIG_TYPES` 且首字母 ∈
         `_TYPE_PREFIX[type]`（与 desig 通道同一套 type/首字母过滤）。

    只改 `cls`；返回逐条提升记录（诊断/审核用），字段与文本对象一一对应。

    ⚠ 记录里的 `trigger_*` 是**校正资格的证据**（"谁让它够格被提升"），**不是最终匹配对象**：
    这里命中第一个合格器件就 `break`，而随后 `match()` 的 desig 通道会按距离做**双向唯一**
    重分配，最终可能落到另一个器件、甚至不落（见 `tests/test_desig_reclass.py::
    TestTriggerVsFinalMatch`）。要看最终关联，请读交付 JSON 的 `key`。
    **不要为改这条日志去改选择逻辑。**
    """
    out: list[dict] = []
    for ti, t in enumerate(texts):
        if getattr(t, "cls", None) != "text_other":
            continue
        s = _desig_norm(t.content)
        if _is_numeric_text(s):
            continue
        c = norm(s)
        if not _DESIG_RE.fullmatch(c):
            continue
        tc = t.center()
        hit = hit_d = None
        for i, cp in enumerate(comps):
            bb = getattr(cp, "bbox", None)
            if bb is None:
                continue
            ttype = getattr(cp, "ctype", None) or getattr(cp, "type", None) or ""
            if ttype not in _DESIG_TYPES or c[0] not in _TYPE_PREFIX.get(ttype, ()):
                continue
            d = _point_box_dist(tc, bb)
            if d <= MAX_DESIG_DIST:
                hit, hit_d = i, d
                break
        if hit is None:
            continue
        # trigger_comp_* 是**管线内部**的器件下标/框，让审核方能把这条记录按几何 join
        # 回交付 JSON（交付 key 是管线自己编的）；只表示"满足校正资格的触发证据"。
        out.append({"ti": ti, "content": t.content, "old_cls": t.cls,
                    "new_cls": "designator", "bbox": [float(v) for v in t.bbox],
                    "trigger_comp_idx": hit,
                    "trigger_comp_bbox": [float(v) for v in getattr(comps[hit], "bbox")],
                    "trigger_dist": round(float(hit_d), 2)})
        t.cls = "designator"
    return out


def _center(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _bbox_of(cp):
    """DraftComp.bbox（y-up）或 GT Box.bbox_up。"""
    b = getattr(cp, "bbox", None)
    if b is None:
        b = getattr(cp, "bbox_up", None)
    return b


def _pins_of(cp):
    return getattr(cp, "pins", None) or []


def _point_box_dist(p: tuple[float, float], box) -> float:
    """点到轴对齐框（y-up）的欧氏距离；点在框内为 0。"""
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return float((dx * dx + dy * dy) ** 0.5)


def _point_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """点到点欧氏距离（引脚匹配用，pin.point 是 2 元组）。"""
    dx, dy = a[0] - b[0], a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)


@dataclass
class MatchResult:
    desig_text: dict[int, str] = field(default_factory=dict)        # comp_idx → 位号
    pin_numbers: dict[tuple[int, int], str] = field(default_factory=dict)  # (ci, pi) → 脚号
    pinnames: dict[tuple[int, int], str] = field(default_factory=dict)     # (ci, pi) → 引脚名
    values: dict[int, str] = field(default_factory=dict)            # comp_idx → value
    stats: dict = field(default_factory=dict)                       # 计数（调试/评估用）


def _greedy_assign(cands: list[tuple[float, int, str]]) -> dict[int, str]:
    """候选 (dist, idx, content)，按 dist 升序贪心：每个 idx 取最近一次。"""
    out: dict[int, str] = {}
    for dist, idx, content in sorted(cands, key=lambda t: t[0]):
        if idx not in out:
            out[idx] = content
    return out


def _greedy_assign_bidir(cands: list[tuple[float, int | tuple[int, int], int, str]]) -> dict:
    """候选 (dist, idx, text_idx, content)：**双向唯一** —— 一个文本最多赋给
    一个目标（元件或引脚），一个目标最多收一个文本。消除"R5 同时贴给两个
    元件"以及"同一脚号贴给多个引脚"的重复分配。idx 可为 comp 索引或
    (comp_idx, pin_idx) 二元组。"""
    out: dict[int, str] = {}
    used_t: set[int] = set()
    for dist, idx, ti, content in sorted(cands, key=lambda t: t[0]):
        if idx in out or ti in used_t:
            continue
        out[idx] = content
        used_t.add(ti)
    return out


def match(comps: list, texts: list[OcrText]) -> MatchResult:
    """comps：带 y-up 框的对象（DraftComp 有 `.bbox`/`.pins`；GT Box 有 `.bbox_up` 且
    无 `.pins` → 只参与位号/value 匹配，pin 匹配跳过）。texts：OcrEngine 输出。"""
    res = MatchResult()

    # 1) designator → 元件位号（双向唯一）
    # 匹配键是 **content 形态**，不是 1.1e 的类别：designator 类直接用；
    # text_other 里形似位号（R5/C6/U1…）也救回 —— 1.1e 的 designator↔text_other
    # 混淆严重（实测 val 有 6 例 designator=0、位号全被分进 text_other）。
    # pinName（GPI04/SCK 也形似"字母+数字"）不抢 desig 通道。
    # 2026-09-17：**通道放宽到 pinNumber/value** —— c 桶归因实测 7 个"读对但丢"的位号
    # （USB1/U1/R13）被 1.1e 打成 pinNumber/value，进不了 desig 通道；_DESIG_RE 以字母
    # 开头，pinNumber 的纯数字（2/19）与 value 的数值（100nF）天然不匹配，不会误入。
    desig_cands: list[tuple[float, int, int, str]] = []
    _cand_idx: dict[tuple, int] = {}
    for ti, t in enumerate(texts):
        if t.cls not in ("designator", "text_other", "pinNumber", "value"):
            continue
        c = norm(t.content)
        if not _DESIG_RE.match(c):
            continue
        tc = t.center()
        for i, cp in enumerate(comps):
            bb = _bbox_of(cp)
            if bb is None:
                continue
            ttype = getattr(cp, "ctype", None) or getattr(cp, "type", None) or ""
            # 2026-09-17 v6：type 过滤只对"救回通道"生效（text_other/pinNumber/value）——
            # 1.1e 明确判为 designator 的文本是位号的一手证据，信任它（val35 实测：
            # 0170 一批连接器被 YOLO 分成 box，GT 位号是 JP/CH/CN/H/LD/o 等非 U 前缀，
            # 全量过滤把它们误杀，-2.9 分）。
            # v8：type 过滤对所有 cls 生效（含 designator）——gnd/v/net_short/jumper 等
            # 19 个 type 的 GT key 完全不是位号形（probe_type_prefix 实测），即使 1.1e 判
            # 为 designator 的文本也不该给它们（实测 0199/C18 被 ∅12(gnd) 抢走）。box/r/c
            # 等位号型 type 不受影响（0170 连接器 box 位号 JP/LD/CH 仍收）。
            if ttype not in _DESIG_TYPES:
                continue                       # gnd/v/net_short 等不收位号（GT key 非位号形）
            # v11：首字母过滤对所有 cls 生效（含 designator）——v7 已把 box 放宽到
            # UJLOCH（0170 JP1/LD1/CH1/CN2/H1/o1 全在），0106 I5p / 0199 I9 假文本仍被拦；
            # 实测 0108 R11 的 designator 版文本被 amp 元件收（R∉{U} 未拦）致错配。
            if c[0] not in _TYPE_PREFIX.get(ttype, ()):
                continue                       # 位号首字母必须匹配元件 type（杀假文本抢占）
            d = _point_box_dist(tc, bb)
            if d <= MAX_DESIG_DIST:
                # v10：同位置重复文本去重（1.1e 对同一行字常出重叠双框，0163 R2 / 0199 R21
                # 各有 text_other+designator 双副本；dist=50 后双副本各配一个元件导致错配
                # D2→R2、U4→R21）。键=(center, content)：designator 优先，其次距离小者。
                _dk = (round(tc[0]), round(tc[1]), c)
                if _dk in _cand_idx:
                    _cur = _cand_idx[_dk]
                    _old_d, _old_i, _old_ti, _old_c = desig_cands[_cur]
                    _old_t = texts[_old_ti]
                    if (t.cls == "designator" and _old_t.cls != "designator") or (
                            t.cls == _old_t.cls and d < _old_d):
                        desig_cands[_cur] = (d, i, ti, t.content)
                else:
                    _cand_idx[_dk] = len(desig_cands)
                    desig_cands.append((d, i, ti, t.content))
    res.desig_text = _greedy_assign_bidir(desig_cands)

    # 2) pinNumber → 物理脚号（**双向唯一**：一个文本最多赋一个引脚，一个引脚最多
    #    收一个文本——v2 修过的「151>123」重复分配同类风险，pin 通道同样适用）
    pin_cands: list[tuple[float, tuple[int, int], int, str]] = []
    for ti, t in enumerate(texts):
        if t.cls != "pinNumber":
            continue
        c = norm(t.content)
        if not _PIN_RE.match(c):
            continue
        tc = t.center()
        for ci, cp in enumerate(comps):
            for pi, pin in enumerate(_pins_of(cp)):
                d = _point_dist(tc, pin.point)
                if d <= MAX_PIN_DIST:
                    pin_cands.append((d, (ci, pi), ti, t.content))
    res.pin_numbers = _greedy_assign_bidir(pin_cands)

    # 3) pinName → 引脚名（同双向唯一）
    pn_cands: list[tuple[float, tuple[int, int], int, str]] = []
    for ti, t in enumerate(texts):
        if t.cls != "pinName":
            continue
        tc = t.center()
        for ci, cp in enumerate(comps):
            for pi, pin in enumerate(_pins_of(cp)):
                d = _point_dist(tc, pin.point)
                if d <= MAX_PIN_DIST:
                    pn_cands.append((d, (ci, pi), ti, t.content))
    res.pinnames = _greedy_assign_bidir(pn_cands)

    # 4) value → 元件值（双向唯一，**两轮分层**）
    # 2026-09-17 v2：value 通道放宽到全 cls 后出现回归（tp 5→2）——text_other 里
    # 乱读的"value 形"文本（89RNRR/3499/10.0）抢先占元件，挤掉原本正确的
    # cls=value 文本。修复：**第一轮只用 cls=value（高可信）**；第二轮才把
    # text_other 等里形似 value 的补进来，且只补**还没 value 的元件**。
    # 纯脚号（_PIN_RE，如 3B）排除防污染。
    def _value_cands(texts, cls_ok, shape_ok):
        cands: list[tuple[float, int, int, str]] = []
        for ti, t in enumerate(texts):
            if t.cls not in cls_ok:
                continue
            c = norm(t.content)
            if shape_ok and not (_VALUE_RE.match(c) or _NETNAME_RE.match(c)):
                continue
            tc = t.center()
            for i, cp in enumerate(comps):
                bb = _bbox_of(cp)
                if bb is None:
                    continue
                d = _point_box_dist(tc, bb)
                if d <= MAX_VALUE_DIST:
                    cands.append((d, i, ti, t.content))
        return cands

    # 第一轮：cls=value 全收（含纯数字 value 如 100）——恢复原版行为，0 回归
    res.values = _greedy_assign_bidir(_value_cands(texts, ("value",), shape_ok=False))
    # 第二轮：其他 cls 的 value 形文本（_VALUE_RE 形态过滤防乱读），只补没 value 的元件
    have_val = set(res.values)
    extra = [(d, i, ti, c) for d, i, ti, c in _value_cands(texts, ("text_other", "netlabel"), shape_ok=True)
             if i not in have_val]
    res.values.update(_greedy_assign_bidir(extra))

    res.stats = {
        "n_texts": len(texts),
        "n_desig": len(res.desig_text),
        "n_pin_num": len(res.pin_numbers),
        "n_pinname": len(res.pinnames),
        "n_value": len(res.values),
    }
    return res


def apply_to_draft(draft, res: MatchResult) -> None:
    """把 MatchResult 覆写到 DraftCase（原位修改）。"""
    for ci, content in res.desig_text.items():
        if 0 <= ci < len(draft.comps):
            draft.comps[ci].desig_text = content
            draft.comps[ci].desig_conf = 0.5          # 预测位号，非 GT，置信度降低
    for ci, content in res.values.items():
        if 0 <= ci < len(draft.comps):
            draft.comps[ci].value = content
    for (ci, pi), content in res.pin_numbers.items():
        if 0 <= ci < len(draft.comps) and 0 <= pi < len(draft.comps[ci].pins):
            draft.comps[ci].pins[pi].pin_number_label = content
    for (ci, pi), content in res.pinnames.items():
        if 0 <= ci < len(draft.comps) and 0 <= pi < len(draft.comps[ci].pins):
            draft.comps[ci].pins[pi].pinname = content
