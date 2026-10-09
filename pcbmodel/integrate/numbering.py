# -*- coding: utf-8 -*-
"""P1 阶段 0.5.2：编号生成器（决策 14）—— **可切换 policy**。

**要解决的问题**：官方 component/pin 都是**严格 key** 口径，key 错了内容再对也白搭。
但 key 的可学性**分两支**（§零 归因分析 1）：
  - **命名元件**：`pin_<n>` = 图上的物理脚号（实测 **80.5%** 恰为 `{1..n}`）→ **要当学习目标**，
    优先用 OCR 读到的 `pinNumber` 文本，退化时按几何序枚举。
  - **∅ 元件**（`gnd`/`v`/`net_*` 等电网符号）：位号与 pin 号同出源文档的一个**内部计数器**，
    **物理上不可复现**（实测 6 种重建模型 835 样本仅命中 1 次）→ **不当学习目标**。
    这里只负责产出**唯一、自洽、确定**的键，使 JSON 合法、几何可配对；
    **分数靠官方决策 17 的放宽**（`gnd`/`v` 的序号官方不比对）。
    ⚠ **别在 ∅ 编号上做规则挖掘** —— 实测「同前缀按几何序」只解释 **14.1%**（随机 baseline≈0%），
    且 `∅ = max(pin)+2` 与 `+3` 是**五五开**（50.5% / 49.3%），不是稳定规律。

**为什么 policy 要可切换**：编号是**假设**不是事实（官方计数器不可见）。与其辩论哪种对，
不如在 golden 10 例上把每种假设都扫一遍，**取实测高分者**。

⚠ 全流程**顺序不可换**：`编号 → 净化 → 打分`。净化器**绝不重新编号**（它只做格式修复）。

坐标：官方 target 为 **y-up 原点左下**；"阅读序"= 上→下、左→右（实测最能解释编号的几何序）。

运行（pcbeval env，项目根）：
  python -m pcbmodel.integrate.numbering --sweep        # golden 10 例上扫全部 policy
  python -m pcbmodel.integrate.numbering --case 0025    # 单例编号结果
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.config import PIN_NUM_ABS_FLOOR, PIN_NUM_REL_SLACK
from pcbeval.refs import VOID, normalize_void

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "stage05"

# 官方词表（小写） → 位号前缀。注意这与 `pcbeval.config.AUTO_PREFIX` 的 P0 词表**不同**，
# 官方词表是 box/r/c/l/... 而非 ic/resistor/capacitor/...，不可混用。
OFFICIAL_PREFIX: dict[str, str] = {
    "r": "R", "c": "C", "l": "L", "d": "D", "led": "D", "fuse": "F",
    "box": "U", "switch": "SW", "testpoint": "TP", "pin": "TP",
    "crystal_2pin": "Y", "crystal_3pin": "Y", "crystal_4pin": "Y", "crystal": "Y",
    "mosfet": "Q", "mosfet_npn": "Q", "mosfet_pnp": "Q",
    "bjt": "Q", "bjt_npn": "Q", "bjt_pnp": "Q",
    "amp": "U", "amp_3pin": "U", "amp_5pin": "U",
    "block": "BLK", "dc": "DC", "esd": "ESD", "jumper": "JP",
    "opt": "OC", "seg": "SEG", "ant": "ANT", "buzzer": "BZ",
    "motor": "M", "battary": "BT", "transfomer": "T",
    "circle_header": "J", "m3螺丝": "MH", "other": "X",
}

# ∅ 位号的类型集合 —— **实测得出，不是猜的**（全 210 例扫描）：
# 下列 19 类 **100% 是 ∅**，另加 `v`（1102/1103 = 99.9%）与 `amp` 族（80%/27%/20%，取多数）。
# 其余类型（`r`/`c`/`box`/`d`/`led`/`l`/`switch`/`crystal*`…）**从不出现在 ∅ 里**。
void_types_measured: dict[str, tuple[int, int]] = {
    "gnd": (2170, 2170), "v": (1102, 1103), "net_input": (339, 339),
    "net_short": (163, 163), "net_output": (76, 76), "other": (69, 69),
    "m3螺丝": (58, 58), "battary": (24, 24), "buzzer": (21, 21),
    "block": (13, 13), "net_bidirection": (13, 13), "jumper": (9, 9),
    "motor": (8, 8), "amp": (8, 10), "esd": (6, 6), "dc": (6, 6),
    "transfomer": (5, 5), "circle_header": (4, 4), "ant": (4, 4),
    "amp_5pin": (3, 11), "amp_3pin": (2, 10), "opt": (2, 2), "seg": (1, 1),
}
# 阈值取多数：`amp`(80%) 进；`amp_5pin`(27%)/`amp_3pin`(20%) **不进**（多数是命名件）
VOID_TYPES: frozenset[str] = frozenset(
    t for t, (v, n) in void_types_measured.items() if v / n >= 0.5)


# --------------------------------------------------------------------------- #
# 输入草案（检测/OCR 阶段的产物；本轮由 runner 从 GT 构造）
# --------------------------------------------------------------------------- #
@dataclass
class DraftPin:
    point: tuple[float, float]
    pin_number_label: str | None = None     # OCR 读到的物理脚号
    pinname: str | None = None
    conf: float = 1.0


@dataclass
class DraftComp:
    bbox: list[float]                        # y-up
    ctype: str
    name: str | None = None
    value: str | None = None
    desig_text: str | None = None            # OCR 读到的位号（None = 没读到）
    desig_conf: float = 1.0
    pins: list[DraftPin] = field(default_factory=list)


@dataclass
class DraftNet:
    members: list[tuple[int, int]] = field(default_factory=list)   # (comp_idx, pin_idx)
    label: str | None = None
    edges: list[list[dict]] = field(default_factory=list)          # 每段恰 2 点


@dataclass
class DraftCase:
    case_id: str
    W: int
    H: int
    comps: list[DraftComp] = field(default_factory=list)
    nets: list[DraftNet] = field(default_factory=list)
    # 仅 runner 的 GT 路径用：记录草案 ↔ GT 的位置对应，便于把 GT net 的 token
    # 映射成 (comp_idx, pin_idx)。生产路径（detect）不需要这两个字段。
    gt_desigs: list[str] = field(default_factory=list)
    gt_pin_keys: list[list[str]] = field(default_factory=list)
    # box 引脚链的输入：本例 OCR 的**全部类**文本（`OcrText`，y-up）。
    # ⚠ 只有 `ocr-real` 路径会填；其余 predictor 留空 ⇒ `box_pins` 自动不生效（逐位不变）。
    # ⚠ 不许按 `cls=='pinName'` 过滤（跳 0.5：数字脚名的类纯度只有 3.5%），见 `box_pins`。
    box_texts: list = field(default_factory=list)


@dataclass(frozen=True)
class NumberingPolicy:
    name: str
    pin_source: str = "ocr_then_geom"   # ocr_then_geom | ocr_only | geom_only
    use_ocr_desig: bool = True          # 命名元件是否采信 OCR 位号
    # `u2`/`U2` 大小写归一。**默认已翻成 False**（2026-09-17）—— 一手依据：GT 保源图大小写
    # （`u1`/`c13`/`RvC5`，全集 22 例 / 40 个位号），而图上丝印与 GT key **逐字（含大小写）
    # 相同 3002/3003** ⇒ 归一化是在**制造**错误。
    # ⚠ 但**别指望翻开它能加分**：真实 OCR 档实测 **Δ = 0.000**（`ocr-real`+GT 框，val19
    #   27.270 → 27.270）—— CRNN **自己就把 `u2` 读成 `U2`**（`runs/stage05/recog_gap.md`），
    #   这个开关在真管线上**根本没机会触发**。它在 `ocr-sim`/`oracle-keys` 上值 +1.99/+2.13，
    #   那两个档喂的是 GT 原文（大小写一定对）⇒ **是 harness 假象，不是可交付收益**。
    #   真正的病在识别器（小写盲：`CvC1`→`CC1`、`RvC2`→`ReC2`），归 OCR 线。
    #   翻 False 的理由不是"现在能赚"，是：① 交付路径实测零影响；② 识别器一旦学会大小写，
    #   True 会**主动吃掉**那份收益；③ 它让 `ocr-sim` 那类 harness 不再系统性低估。
    allow_uppercase_fix: bool = False
    void_mode: str = "seq"              # seq | maxpin
    void_delta: int = 2                 # maxpin 模式的 +2 / +3
    order: str = "reading"              # reading | x_then_y | source
    dup_suffix: bool = True             # 官方 `pin_<x>_1`/`_2` 重号规则
    net_name: str = "net_seq"           # net_seq | label_then_seq
    gt_only: bool = False               # ⚠ 吃了 GT 信息才有收益 ⇒ 不是可选 policy，见 void_mode=maxpin


POLICIES: dict[str, NumberingPolicy] = {
    "default":    NumberingPolicy("default"),
    "ocr_only":   NumberingPolicy("ocr_only", pin_source="ocr_only"),
    "geom_only":  NumberingPolicy("geom_only", pin_source="geom_only", use_ocr_desig=False),
    "void_delta3": NumberingPolicy("void_delta3", void_mode="maxpin", void_delta=3,
                                   gt_only=True),
    "void_seq":   NumberingPolicy("void_seq"),
    "no_dup_suffix": NumberingPolicy("no_dup_suffix", dup_suffix=False),
    "x_order":    NumberingPolicy("x_order", order="x_then_y"),
}


# --------------------------------------------------------------------------- #
# 排序与守卫
# --------------------------------------------------------------------------- #
def _key_rl(c: DraftComp):
    x1, y1, x2, y2 = c.bbox
    return (-(y1 + y2) / 2.0, (x1 + x2) / 2.0)      # y-up：上→下、左→右


def _key_x(c: DraftComp):
    x1, y1, x2, y2 = c.bbox
    return ((x1 + x2) / 2.0, -(y1 + y2) / 2.0)


def _order(comps: list[DraftComp], mode: str) -> list[int]:
    if mode == "source":
        return list(range(len(comps)))
    k = _key_x if mode == "x_then_y" else _key_rl
    return sorted(range(len(comps)), key=lambda i: k(comps[i]))


def pin_number_acceptable(label: str, n_pins: int) -> bool:
    """OCR 纯数字脚号的合理性守卫（复用 `config` 的双保险阈值）。

    官方附录一：「pinNumber 为空、∅ 占位符或 OCR 噪声时自动编号」。真脚数不可知，
    故用 `max(REL_SLACK × 标注脚数, ABS_FLOOR)` 作宽松上限，拦掉 `pin_99421` 这类垃圾号。
    """
    if not label or not label.isdigit():
        return bool(label)              # 字母脚号（A5/EP）一律接受
    n = int(label)
    if n <= 0:
        return False
    return n <= max(PIN_NUM_REL_SLACK * max(n_pins, 1), PIN_NUM_ABS_FLOOR)


# --------------------------------------------------------------------------- #
# 编号
# --------------------------------------------------------------------------- #
def assign_designators(draft: DraftCase, policy: NumberingPolicy) -> list[str]:
    """给每个元件分配位号。返回与 `draft.comps` 等长的列表。"""
    n = len(draft.comps)
    out: list[str | None] = [None] * n
    used: set[str] = set()
    counters: dict[str, int] = {}

    # ① 命名元件：**两趟** —— OCR 位号先全发掉，再给剩下的自动编号。
    order = _order(draft.comps, policy.order)

    def _voidish(i: int) -> bool:
        """∅ 元件（走 ② 的自洽编号）。⚠ ∅-前缀的 desig_text **不算**可用的位号：它来自
        那个不可复现的源文档计数器，照抄没意义（再怎么抄也对不上 GT）。"""
        c = draft.comps[i]
        return (c.ctype in VOID_TYPES
                or normalize_void((c.desig_text or "").strip()).startswith(VOID))

    def _bump(base: str) -> str:
        """撞车时加后缀。⚠ 必须 `base_2 / base_3 …` 递增 —— 写成 `cand += "_2"` 的循环
        会产出 `R4_2_2`（三次撞车时），GT 里两个都不存在。"""
        n = 1
        cand = base
        while cand in used:
            n += 1
            cand = f"{base}_{n}"
        return cand

    # ①a **先把 OCR 读到的位号全部发掉** —— 一手证据优先，别让自动号抢在前面。
    # ⚠⚠ **这一趟不是优化、是修 bug，顺序不可换回「一趟走完」**：旧版一趟走时，
    #   阅读序里**先碰上的、图上没印字的元件**会先拿到自动号 `R4`，随后**真 R4**
    #   （字读到了）晚一步撞车、被改名 `R4_2` —— 而 GT 里没有 `_2`，**两个都白丢**。
    #   实测 val19 ocr-sim：**17 处 `_2` 全部是这一类**（自动号先占真位号），
    #   **无一例**是「两张 OCR 文本归一后同名」、也**无一例**来自大小写归一
    #   （后者是另一条独立的 bug，见 `allow_uppercase_fix`）。
    #   ⇒ 不需要「谁离位号文本近谁保原名」那套几何仲裁：自动号让路即可，17 → 0。
    for i in order:
        if _voidish(i):
            continue
        c = draft.comps[i]
        dt = normalize_void((c.desig_text or "").strip())
        if policy.use_ocr_desig and dt and c.desig_conf > 0 and not dt.endswith("?"):
            # `U?`/`R?` = 图上未标注，不可当位号用（落到 ①b 走自动号）
            cand = _bump(dt.upper() if policy.allow_uppercase_fix else dt)
            used.add(cand)
            out[i] = cand

    # ①b 再给没读到位号的元件按几何序自动编号，**计数器跳过已被 OCR 占用的名字**。
    for i in order:
        if _voidish(i) or out[i] is not None:
            continue
        pfx = OFFICIAL_PREFIX.get(draft.comps[i].ctype, "X")
        k = counters.get(pfx, 0) + 1
        while f"{pfx}{k}" in used:
            k += 1
        counters[pfx] = k
        cand = f"{pfx}{k}"
        used.add(cand)
        out[i] = cand

    # ② ∅ 元件：只求唯一 + 自洽（分数靠官方放宽，见模块 docstring）
    vidx = [i for i in _order(draft.comps, policy.order)
            if draft.comps[i].ctype in VOID_TYPES
            or normalize_void((draft.comps[i].desig_text or "").strip()).startswith(VOID)]
    if policy.void_mode == "maxpin":
        # ⚠⚠ **这是一个 oracle，不是可选 policy。别在真实链路上用它。**
        # 实测（2026-09-11，val 19 例）：`∅号 = 该元件最大纯数字 pin 键号 + 2 或 3`，
        # 覆盖 **99.9%**（3964 个 ∅ 元件：delta=2 占 51.8%、=3 占 48.1%、其余 4 个）。
        # 但那个 pin 键号是**官方 target 生成期的全局计数器**，图上没印、源 CVJsonStd 里
        # 也没有（`∅580` 是个 3.3V 电源符号，其 pin `pin_578` 在源文件里
        # `pinNumberByOCR`/`pinNumberByVLM` 全是 null；裁图目视确认图上无 `578`）。
        # → 它只在 `draft_from_gt(with_pin_numbers=True)` 这种吃了 GT pin 键的草案上有效：
        #   **gt-boxes 上 compF1 0.1628 → 0.4313，detect 上收益恰好为 0（0.0443 不变）**。
        # 谁要是看 `--sweep` 的排行榜选了这个 policy，就是选了一个偷答案的。

        for i in vidx:
            nums = [int(p.pin_number_label) for p in draft.comps[i].pins
                    if p.pin_number_label and p.pin_number_label.isdigit()]
            base = (max(nums) if nums else 0) + policy.void_delta
            cand = f"{VOID}{base}"
            while cand in used:
                base += 1
                cand = f"{VOID}{base}"
            used.add(cand)
            out[i] = cand
    else:
        for k, i in enumerate(vidx, 1):
            cand = f"{VOID}{k}"
            while cand in used:
                k += 1
                cand = f"{VOID}{k}"
            used.add(cand)
            out[i] = cand

    return [o if o is not None else f"X{i + 1}" for i, o in enumerate(out)]


def assign_pin_keys(comp: DraftComp, policy: NumberingPolicy) -> list[str]:
    """给一个元件的引脚分配 `pin_<n>`。顺序与 `comp.pins` 一致。"""
    n = len(comp.pins)
    keys: list[str | None] = [None] * n

    if policy.pin_source != "geom_only":
        for i, p in enumerate(comp.pins):
            lb = (p.pin_number_label or "").strip()
            if lb and pin_number_acceptable(lb, n):
                keys[i] = f"pin_{lb}"

    if policy.pin_source != "ocr_only":
        # 几何序回填：竖直 → 下→上；水平 → 左→右（与 pin_geom 的 pin_1 口径一致）
        xs = [p.point[0] for p in comp.pins]
        ys = [p.point[1] for p in comp.pins]
        x1, y1, x2, y2 = comp.bbox
        vert = (y2 - y1) >= (x2 - x1)
        order = sorted(range(n), key=lambda i: ys[i] if vert else xs[i])
        nxt = 1
        for i in order:
            if keys[i] is None:
                while f"pin_{nxt}" in keys:
                    nxt += 1
                keys[i] = f"pin_{nxt}"
                nxt += 1

    for i in range(n):
        if keys[i] is None:
            keys[i] = f"pin_{i + 1}"

    if policy.dup_suffix:
        total: dict[str, int] = {}
        for k in keys:
            total[k] = total.get(k, 0) + 1
        seen: dict[str, int] = {}
        for i, k in enumerate(keys):
            if total[k] > 1:
                # 官方重号规则：同名脚 **全部**加 1-based 后缀 → pin_19_1 / pin_19_2
                seen[k] = seen.get(k, 0) + 1
                keys[i] = f"{k}_{seen[k]}"
    return keys                                                    # type: ignore[return-value]


def assign_net_names(draft: DraftCase, policy: NumberingPolicy) -> list[str]:
    """`net_<n>` 兜底名。⚠ 官方明文 **net key 不参与匹配**，不必花力气对齐。"""
    out: list[str] = []
    used: set[str] = set()
    for i, net in enumerate(draft.nets, 1):
        nm = None
        if policy.net_name == "label_then_seq" and net.label:
            nm = net.label
        if not nm:
            nm = f"net_{i}"
        while nm in used:
            nm = f"{nm}_2"
        used.add(nm)
        out.append(nm)
    return out


def apply(draft: DraftCase, policy: NumberingPolicy | None = None) -> dict:
    """草案 → 官方 target dict（**尚未净化**；顺序必须是 编号 → 净化 → 打分）。"""
    policy = policy or POLICIES["default"]
    desigs = assign_designators(draft, policy)
    pin_keys = [assign_pin_keys(c, policy) for c in draft.comps]
    net_names = assign_net_names(draft, policy)

    components: dict = {}
    pins: dict = {}
    for i, c in enumerate(draft.comps):
        d = desigs[i]
        components[d] = {"Name": c.name or d, "type": c.ctype,
                         "value": c.value, "bbox": [float(v) for v in c.bbox]}
        pd: dict = {}
        for j, p in enumerate(c.pins):
            pd[pin_keys[i][j]] = {
                "pinname": p.pinname if p.pinname is not None else pin_keys[i][j][4:],
                "point": {"x": float(p.point[0]), "y": float(p.point[1])}}
        pins[d] = pd

    nets: dict = {}
    for ni, net in enumerate(draft.nets):
        toks = []
        for (ci, pi) in net.members:
            if ci >= len(desigs) or pi >= len(pin_keys[ci]):
                continue
            toks.append(f"{desigs[ci]}.{pin_keys[ci][pi][4:]}")
        edges = {}
        for ei, seg in enumerate(net.edges, 1):
            edges[f"edge_{ei}"] = [{"x": float(pt["x"]), "y": float(pt["y"])} for pt in seg]
        nets[net_names[ni]] = {"hyperGraph": "(" + ",".join(toks) + ")", "edges": edges}
    return {"components": components, "pins": pins, "nets": nets}


def render_md(rows: list[dict], title: str = "编号 policy 扫描") -> str:
    L = [f"# {title}", "",
         "| policy | 单例分 | compF1 | pinF1 | netF1 | lineF1 |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: -x.get("score", 0)):
        mark = " ⚠oracle" if r.get("gt_only") else ""
        L.append(f"| `{r['policy']}`{mark} | **{r.get('score', 0):.2f}** | {r.get('comp', 0):.4f} | "
                 f"{r.get('pin', 0):.4f} | {r.get('net', 0):.4f} | {r.get('line', 0):.4f} |")
    L += ["", "> 口径：官方 golden 10 例，strict（交付口径）。"
              "编号是**假设**不是事实（源计数器不可见）→ 以实测高分者为准，不靠论证。",
          "> ⚠ **`gt_only` 标了 oracle 的行不能用**：它只在吃了 GT pin 键的草案上有收益，"
          "真实链路（`detect`）收益为 0。它必然排第一，**别选它**。", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="numbering", description="编号生成器（可切换 policy）")
    ap.add_argument("--sweep", action="store_true", help="golden 10 例上扫全部 policy")
    ap.add_argument("--case", default=None)
    ap.add_argument("--policy", default="default", choices=sorted(POLICIES))
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)

    if a.sweep:
        from pcbmodel.integrate.runner import run_and_score_case
        rows = []
        for name, pol in sorted(POLICIES.items()):
            tot = {"score": 0.0, "comp": 0.0, "pin": 0.0, "net": 0.0, "line": 0.0}
            n = 0
            for r in run_and_score_case(split="golden", policy=pol):
                if r.get("score") is None:
                    continue
                n += 1
                tot["score"] += r["score"]
                for k in ("comp", "pin", "net", "line"):
                    tot[k] += r.get(k, 0.0)
            if n:
                rows.append({"policy": name, "gt_only": pol.gt_only,
                             **{k: v / n for k, v in tot.items()}})
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "numbering_sweep.md").write_text(render_md(rows), encoding="utf-8")
        print(render_md(rows))
        # ⚠ oracle policy 一律排除在「最优」之外 —— 它必然排第一，选了就是偷答案
        fair = [r for r in rows if not r.get("gt_only")]
        if fair:
            best = max(fair, key=lambda x: x["score"])
            print(f"[numbering] 最优 policy = `{best['policy']}`（{best['score']:.2f}，已排除 oracle）"
                  f" -> {out / 'numbering_sweep.md'}")
        return 0

    if a.case:
        from pcbmodel.integrate.runner import draft_from_gt
        d = draft_from_gt(a.case)
        t = apply(d, POLICIES[a.policy])
        print(f"[numbering] {a.case} policy={a.policy}")
        for k, v in t["components"].items():
            print(f"   {k:10s} {v['type']:14s} bbox={v['bbox']}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
