# -*- coding: utf-8 -*-
"""CVJsonStd -> 赛题 target 格式 GT（components / pins / nets）构建。

真值优先级（位号 / 名称 / 值 / pin 编号）：
  designator : .net "待人工确认 ... designator=.." > 关联 designator 文本 > 类型前缀自动编号
  name       : comp.searchNameByOCR -> texts（清洗 ∅/scenario 噪声）
  value      : 关联 value 文本（∅ 等噪声视为无）
  pin number : comp 内按 pid 序，OCR 编号(纯数字)优先；缺失/冲突/超出物理合理上限
               (见 _pin_num_plausible/config.PIN_NUM_*) 回退最小未用正整数 —— 保证唯一；
               官方附录:pinNumber 为 OCR 噪声时自动编号，故不合理纯数字不采信
  pin name   : pinNameByOCR -> texts
  point      : pin.tip

位号分配统一按 component ID 自然排序（cpntId：C_1,C_2,…，官方附录口径）；重复文本位号 -> base_1..base_k，
自动位号按类型前缀+该序分配。type 词表为本地 best-effort，待官方样例校准（见 config.EvalConfig 注释）。

坐标：CVJsonStd 与 PNG 同为左上原点 y 向下；target 需左下原点 y 向上。
y 翻转为等距变换，本地 F1 不受影响。gtbuilder 内部一律存源坐标(y 向下)，
仅在 emit target 时按 cfg.flip_y（步骤 8 起默认 True）翻成 **canonical y-up**
（与上交格式同一套）。viz 读 canonical target 时用 flip_back 翻回 PNG 帧画墨迹。

nets：本文件只解析 .net 中与“元件/位号”相关的行（待人工确认）；
net 分组与线段属步骤 3（Oracle A / segments）。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .caseio import discover_case_files, load_cvjsonstd
from .config import (AUTO_PREFIX, TYPE_MAP, EvalConfig,
                     PIN_NUM_REL_SLACK, PIN_NUM_ABS_FLOOR)
from . import nets as _nets

# OCR/文本噪声：作为“无内容”处理
_NOISE_MARK = ("∅", "请勿更新库", "后续请勿更新库")
_DESIGNATOR_RE = re.compile(r"待人工确认:\s*(C_\d+)\s+designator=\"([^\"]*)\"")
_NUM_RE = re.compile(r"^\s*(\d+)\s*$")


def is_noise(s: str | None) -> bool:
    if s is None:
        return True
    s = s.strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return True
    if s.startswith(_NOISE_MARK):
        return True
    return False


def _name_ok(s: str | None) -> bool:
    if is_noise(s):
        return False
    low = s.lower()
    return not ("scenario" in low or "original" in low)


@dataclass
class PinRec:
    pid: str            # P_<n>（CVJsonStd 主登记键）
    num: int            # 最终 pin_<n> 的 n（comp 内唯一）
    pinname: str | None
    tip: list           # [x, y] 源坐标(y 向下)
    edge_id: str | None

    @property
    def key(self) -> str:
        return f"pin_{self.num}"


@dataclass
class CompRec:
    cpnt_id: str            # C_<n>
    category: str
    ctype: str              # 目标 type（TYPE_MAP）
    bbox: list              # [x1,y1,x2,y2] 源坐标
    name: str | None = None
    value: str | None = None
    designator_text: str | None = None   # 关联文本(designator)桶
    designator: str | None = None
    raw_pins: dict = field(default_factory=dict)     # comp['pins'] 原始 dict
    pins: list = field(default_factory=list)         # list[PinRec]（pid 序）

    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _resolve(tid, texts: dict) -> str | None:
    """CVJsonStd 里 OCR 类字段多为 text id(T_xx)，解析回 content。"""
    if isinstance(tid, str) and tid in texts:
        v = texts[tid].get("content")
        return v if isinstance(v, str) else None
    return None


def _read_net_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _net_designator_hints(net_text: str) -> dict[str, str]:
    """解析 .net “待人工确认: C_k designator=..” 行 -> {cpntId: designator}。"""
    out: dict[str, str] = {}
    for m in _DESIGNATOR_RE.finditer(net_text):
        cid, ds = m.group(1), m.group(2).strip()
        if not is_noise(ds):
            out[cid] = ds
    return out


class GtBuilder:
    """把单个 case 的 CVJsonStd(+ .net 位号提示) 转成赛题 target-GT。"""

    def __init__(self, cfg: EvalConfig | None = None):
        self.cfg = cfg or EvalConfig()
        self.nets: dict = {}
        self.net_meta: dict = {}

    # ---- 组件收集 ------------------------------------------------------
    def _collect_comps(self) -> list[CompRec]:
        texts = self.cv["texts"]
        comps: list[CompRec] = []
        for cid, c in self.cv["components"].items():
            assoc = c.get("associatedTexts") or {}
            designator_text = value_text = None
            for t in assoc.values():
                det = t.get("details", "")
                content = t.get("content")
                if not isinstance(content, str):
                    continue
                if det == "designator" and not is_noise(content):
                    designator_text = content
                elif det == "value" and not is_noise(content):
                    value_text = content
            name = _resolve(c.get("searchNameByOCR"), texts)
            name = name if _name_ok(name) else None
            bbox = list(c["location"].values()) if isinstance(c.get("location"), dict) else []
            comps.append(CompRec(
                cpnt_id=cid, category=c.get("category", ""),
                ctype=TYPE_MAP.get(c.get("category", ""), "other"),
                bbox=bbox, name=name, value=value_text,
                designator_text=designator_text,
            ))
        return comps

    # ---- 位号分配 ------------------------------------------------------
    def _assign_designators(self, comps: list[CompRec], hints: dict[str, str]) -> None:
        """分配唯一位号。重复文本位号 -> base_1..base_k。

        官方附录(problem6-2026.pdf)：重复 designator 编号顺序按 component ID 自然排序、
        仅出现一次保持原名、适用于所有 type。自动位号(无文本)同样按 cpntId 序分配，
        与官方口径一致（此前用阅读序，已废弃）。
        """
        order = sorted(range(len(comps)),
                       key=lambda i: int(comps[i].cpnt_id.split("_")[1]))
        raw: list[str | None] = []
        for c in comps:
            d = hints.get(c.cpnt_id) or c.designator_text
            raw.append(d.strip() if d is not None else None)
        groups = Counter(d for d in raw if d)
        # 预保留文本位号占用的名字：单例保 base；重复组 base -> base_1..base_k
        reserved: set[str] = set()
        for base, k in groups.items():
            reserved.update([base] if k == 1 else [f"{base}_{j}" for j in range(1, k + 1)])
        used: set[str] = set()
        base_cnt: dict[str, int] = {}
        prefix_cnt: dict[str, int] = {}
        for i in order:
            d = raw[i]
            c = comps[i]
            if d is not None:
                if groups[d] == 1:
                    name = d
                else:  # 重复组内按阅读序取下一个 base_j
                    j = base_cnt.get(d, 0) + 1
                    name = f"{d}_{j}"
                    base_cnt[d] = j
            else:
                prefix = AUTO_PREFIX.get(c.ctype, "X")
                n = prefix_cnt.get(prefix, 0) + 1
                while f"{prefix}{n}" in reserved or f"{prefix}{n}" in used:
                    n += 1
                prefix_cnt[prefix] = n
                name = f"{prefix}{n}"
            used.add(name)
            c.designator = name

    # ---- pin 编号 ------------------------------------------------------
    @staticmethod
    def _pin_num_plausible(cand: int, n_annot: int) -> bool:
        """OCR 纯数字是否像真 pin number（步骤 9 守卫）。

        官方附录：pinNumber 为空/∅/OCR 噪声时自动编号回填；Pin 评分先 component key
        再按 pin number，垃圾号会同时毁 Pin key 与 net hyperGraph token 对官方的对齐。
        真脚数不可知且可能漏标注（0119 U1 真 14 号 > 标注 13），故上限取
        max(PIN_NUM_REL_SLACK × 标注脚数, PIN_NUM_ABS_FLOOR)，只拦远超物理可能的号。
        """
        return cand <= max(PIN_NUM_REL_SLACK * n_annot, PIN_NUM_ABS_FLOOR)

    def _assign_pins(self, comps: list[CompRec]) -> None:
        texts = self.cv["texts"]
        for c in comps:
            raw_pins = sorted(c.raw_pins.items(),
                              key=lambda kv: int(kv[0].split("_")[1]))
            n_annot = len(raw_pins)
            used: set[int] = set()
            recs: list[PinRec] = []
            for pid, pi in raw_pins:
                num = None
                mt = _resolve(pi.get("pinNumberByOCR"), texts)
                if isinstance(mt, str):
                    m = _NUM_RE.match(mt.strip())
                    if m:
                        cand = int(m.group(1))
                        if (cand > 0 and cand not in used
                                and self._pin_num_plausible(cand, n_annot)):
                            num = cand
                if num is None:          # 缺失/冲突/不合理 -> 最小未用正整数
                    n = 1
                    while n in used:
                        n += 1
                    num = n
                used.add(num)
                pinname = _resolve(pi.get("pinNameByOCR"), texts)
                pinname = None if is_noise(pinname) else pinname
                tip = pi.get("tip") or [0.0, 0.0]
                recs.append(PinRec(pid=pid, num=num, pinname=pinname,
                                   tip=[float(v) for v in tip],
                                   edge_id=pi.get("edgeId")))
            c.pins = recs

    # ---- 入口 ----------------------------------------------------------
    def build(self, case_id: str) -> dict:
        self.case_id = case_id
        self.cf = discover_case_files(case_id, self.cfg)
        self.cv = load_cvjsonstd(self.cf.cvjson)
        self.H = self.cv["canvas"]["height"]
        net_text = _read_net_text(self.cf.net)
        hints = _net_designator_hints(net_text)

        comps = self._collect_comps()
        for c in comps:
            c.raw_pins = self.cv["components"][c.cpnt_id].get("pins", {})
        self._assign_designators(comps, hints)
        self._assign_pins(comps)
        self.comps = comps
        self.nets, self.net_meta = _nets.aggregate(
            self.cfg, self.cv, self.comps,
            _nets.parse_net_comments(net_text))
        return self.to_target()

    # ---- 输出（target 格式） ------------------------------------------
    def _tf(self, pt) -> list:
        x, y = float(pt[0]), float(pt[1])
        if self.cfg.flip_y:
            y = self.H - y
        return [x, y]

    def _tf_bbox(self, box) -> list:
        x1, y1, x2, y2 = (float(v) for v in box)
        if self.cfg.flip_y:
            y1, y2 = self.H - y2, self.H - y1
        return [x1, y1, x2, y2]

    def to_target(self) -> dict:
        """按 designator 序输出 components/pins/nets（nets 由 .net Oracle A 构建）。

        步骤 8：① canonical 坐标为 y-up（cfg.flip_y=True，emit 处 y'=H-y），
        与上交格式同一套；② 空 pin 网（hyperGraph "()"，全是 ∅ 噪声 netlabel 回退、
        模型结构上不可预测）从 GT 剔除——登记仍在 net_meta["empty_nets"] 供 WARN/回溯。
        """
        comps = sorted(self.comps, key=lambda c: c.designator)
        components, pins = {}, {}
        for c in comps:
            components[c.designator] = {
                "type": c.ctype, "name": c.name, "value": c.value,
                "bbox": self._tf_bbox(c.bbox),
            }
            pin_d = {}
            for p in sorted(c.pins, key=lambda p: p.num):
                x, y = self._tf(p.tip)
                pin_d[p.key] = {"pinname": p.pinname, "point": {"x": x, "y": y}}
            pins[c.designator] = pin_d
        nets = {nm: n for nm, n in self.nets.items() if n["hyperGraph"] != "()"}
        return {"components": components, "pins": pins, "nets": nets}

    def save(self, case_id: str, out: Path) -> Path:
        self.target = self.build(case_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.target, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        return out
