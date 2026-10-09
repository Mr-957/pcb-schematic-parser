# -*- coding: utf-8 -*-
"""统一配置：路径、阈值、类别映射、坐标/命名开关。单一事实来源。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../code/CV for PCB/


def _default_dataset_root() -> Path:
    return Path(os.environ.get("PCBEVAL_DATASET", PROJECT_ROOT / "Dataset300"))


def _default_out_root() -> Path:
    # 步骤 10：派生结果目录随数据整理收进 prep/；env 仍可覆盖
    return Path(os.environ.get("PCBEVAL_OUT", PROJECT_ROOT / "prep" / "outputs"))


# CVJsonStd component.category -> 赛题输出 type。
# 词表尽量覆盖 Dataset300 实测出现的类别；未知一律 "other"。
# 说明：Gnd->gnd, V->vcc（具体电源轨名由文本区分，未展开 vbus_in；
#       赛题附录把同名测试点/网络标记载为 vbus_in，属待官方校准项）。
TYPE_MAP: dict[str, str] = {
    "Box": "ic",
    "Block": "ic",
    "CIRCLE_HEADER": "connector",
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "Crystal": "crystal",
    "XTAL": "crystal",
    "Switch": "switch",
    "SW": "switch",
    "Jumper": "switch",
    "Connector": "connector",
    "Gnd": "gnd",
    "V": "vcc",
    "Testpoint": "testpoint",
    "NET_input": "netlabel",
    "Net_input": "netlabel",
    "NET_output": "netlabel",
    "Net_output": "netlabel",
    "NET_bidirection": "netlabel",
    "Net_bidirection": "netlabel",
    "Net_short": "netlabel",
    "LED": "led",
    "D": "diode",
    "MOSFET": "mosfet",
    "BJT": "bjt",
    "Fuse": "fuse",
    "M": "motor",
    "Motor": "motor",
    "Buzzer": "buzzer",
    "AMP": "amplifier",
    "ANT": "antenna",
    "Battary": "battery",
    "SEG": "display",
    "PIN": "pin",
    "ESD": "diode",               # ESD/TVS 保护二极管（符号即二极管，位号 D/TVS；待官方词表校准）
    "Dc": "connector",            # DC 电源座/直插座（VIN2/HC-DC-007B 等；待官方词表校准）
    "Transfomer": "transformer",  # 变压器（源标注拼写即 Transfomer）
    "Opt": "optocoupler",         # 4 脚光耦（U 位号；待官方词表校准）
    "Other": "other",
}

# 无 designator 文本时的自动位号前缀（按目标 type）。
AUTO_PREFIX: dict[str, str] = {
    "ic": "U", "resistor": "R", "capacitor": "C", "inductor": "L",
    "crystal": "Y", "switch": "SW", "connector": "J", "gnd": "GND",
    "vcc": "VCC", "netlabel": "NET", "led": "D", "diode": "D",
    "mosfet": "Q", "bjt": "Q", "testpoint": "TP", "other": "X",
    "transformer": "T", "optocoupler": "U",
}

# pin 编号合理性守卫（步骤 9）：OCR 纯数字 pinNumber 只有 ≤ 该器件“物理可能脚数”的
# 宽松上限才被采信，否则视同缺失走自动编号回填 —— 官方附录一：“pinNumber 为空、∅
# 占位符或 OCR 噪声时，自动编号 pin_1…”；Pin 评分先按 component key 再按 pin number。
# 真脚数不可知（标注可能漏脚，0119 U1 真 14 号 > 标注 13 根），故用相对+绝对双保险：
#   上限 = max(PIN_NUM_REL_SLACK × 标注脚数, PIN_NUM_ABS_FLOOR)
# 实测全 300 例下拦掉 124 个 >100 的 OCR 垃圾号（max(3×脚数,100)，52 例），≤100 一律不误伤。
# 具体倍数/下限为猜测，官方噪声判定口径不可见 → 仍待官方样例校准。
PIN_NUM_REL_SLACK = 3
PIN_NUM_ABS_FLOOR = 100

# ---- 官方公开数据集（2026-09-09 到手，权威标注）-----------------------------
# 结构：200_train_cases/<id>/<id>-<tool>_target_named.json（训练标注）
#      10_GTcase/<id>-<tool>/<id>-<tool>_target.json         （golden，标准答案）
# 图片与 Dataset300 对应 case 的 PNG SHA-256 一致（实测 0001..0200）；
# 官方 target 无 CVJsonStd，故 OCR texts / 导线折线等监督仍回源 Dataset300。
OFFICIAL_ROOT = PROJECT_ROOT / "赛题六公开数据集"
OFFICIAL_TRAIN_ROOT = OFFICIAL_ROOT / "200_train_cases"
OFFICIAL_GT_ROOT = OFFICIAL_ROOT / "10_GTcase"

# 元件文本比较字段（把 P0 词表与官方 41–43 类小写词表并起来；GT 该字段为 null 时跳过）。
# 依据：官方 README「IC 比 Name、R/C 比 value」+ HANDOFF §4 value 缺失率（100% 缺 → 比 Name）。
TEXT_NAME_TYPES: set[str] = {
    # P0 词表
    "ic", "connector", "switch", "relay", "mosfet", "bjt", "motor",
    "buzzer", "amplifier", "display", "antenna", "battery", "crystal",
    # 官方词表（box/block=器件框，Name 为型号；其余多为位号/网络名，比 Name 无害）
    "box", "block", "pin", "testpoint", "other", "jumper", "circle_header",
    "dc", "amp", "amp_3pin", "amp_5pin", "mosfet_npn", "mosfet_pnp",
    "bjt_npn", "bjt_pnp", "seg", "opt", "transfomer", "battary", "ant",
    "m3螺丝",
}
TEXT_VALUE_TYPES: set[str] = {
    # P0 词表
    "resistor", "capacitor", "inductor", "fuse", "led", "diode",
    # 官方词表（gnd/v 的 value=网络名；crystal_* 有 value）
    "r", "c", "l", "d", "esd", "gnd", "v",
    "crystal_2pin", "crystal_3pin", "crystal_4pin",
}


def comp_text_field(ctype: str | None) -> str | None:
    """元件该比哪个文本字段：'name' / 'value' / None(不比)。"""
    if ctype in TEXT_NAME_TYPES:
        return "name"
    if ctype in TEXT_VALUE_TYPES:
        return "value"
    return None


def comp_text(c: dict, field: str):
    """读元件文本：兼容官方大写 `Name` 与 P0 小写 `name`。"""
    if field == "name":
        return c.get("Name", c.get("name"))
    return c.get("value")


@dataclass
class EvalConfig:
    """评测相关全部开关与阈值。

    flip_y：坐标方向已定死（步骤 8）——canonical target 一律赛题提交格式
    **左下角 y 向上**，构建时在 emit 处应用 y' = H - y（已验证 flip∘flip=恒等）。
    内部管线（CVJsonStd/PNG/元件 pin tip 等）仍存源坐标 y 向下；viz 画 PNG 时翻回。
    flip_y=True 为唯一规范：本地 GT、评测、上交共用一套 y-up 坐标，不再维护双套。
    """
    dataset_root: Path = field(default_factory=_default_dataset_root)
    out_root: Path = field(default_factory=_default_out_root)

    flip_y: bool = True           # 步骤 8：canonical = y-up（提交格式），y'=H-y
    bbox_tol: float = 20.0
    point_tol: float = 5.0
    line_tol: float = 5.0

    # 字符串比较：官方严格；本地默认“GT pinname 为 null 时跳过比较”（pinname-null 待校准）
    pinname_skip_when_gt_null: bool = True
    # net 匈牙利 cost：交集(推荐) / 并集（官方语义待校准）
    net_cost_mode: str = "intersection"

    # 宽松口径（决策 16/17）。**默认空 = strict 逐字行为不变**（决策 18：可插拔 policy、
    # 不硬编码）。元素 = 特性名，目前只认官方**已确认**的两条：
    #   `"bjt"` → L1 `bjt` 族等价（官方答疑原话，决策 16）
    #   `"gnd"` → L2 `gnd` 位号序号不比对（官方答疑原话，决策 17）
    # ⚠ L3–L5（`gnd_pin`/`v`/`net_*`）官方**未答**（§五 D1），传进来会被忽略、不报错。
    # ⚠ strict 是**交付口径**、loose 是**校准口径**（本地 strict 系统性低估约 7 分）
    #   ⇒ 报告里两个数都要出，**不得混在同一列**。
    loose: frozenset[str] = frozenset()

    # arm3（2026-09-22）：**token 重挂** —— 见 `scorer.align_loose`。默认 False = 逐位不变。
    # ⚠ 这是**口径模拟**、不是提交格式：`loose`（含 `empty`）只把 comp/pin 的 key 改挂到 GT，
    #   hyperGraph token 里的 ∅ 计数器原样保留 ⇒ comp 涨、**net/line 纹丝不动**（三臂 arm2）。
    #   打开本开关后 token 也一起重挂（comp 键按 `ren`、pin 键按**点距 1:1**），net/line 才会动。
    #   语义 = "若官方对 ∅ 元件按坐标配对且 token 跟着重挂"。**收益只在这个口径里成立**，
    #   交付仍报 strict（strict 下补 ∅ 引脚是净负，见 MODEL_HANDOFF 条目 84）。
    token_align: bool = False

    @property
    def gt_dir(self) -> Path:
        return self.out_root / "gt"

    @property
    def reports_dir(self) -> Path:
        return self.out_root / "reports"

    @property
    def viz_dir(self) -> Path:
        return self.out_root / "viz"

    def ensure_dirs(self) -> None:
        for d in (self.gt_dir, self.reports_dir, self.viz_dir):
            d.mkdir(parents=True, exist_ok=True)
