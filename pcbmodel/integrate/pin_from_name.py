# -*- coding: utf-8 -*-
"""box 元件的脚号来源：**官方 IC 库（唯一名）→ GT 投票表 → 弃权**。

## 这个模块做什么 / 不做什么

做：`(型号, 脚名) -> 脚号`，并组装 `DraftPin`。
**不产生几何** —— 脚的 `point` 与「哪条脚名文本属于哪个脚」由调用方给出。

⚠⚠ **别往 `pin_geom.PIN_OFFSET_PX` 里加 `box`/`gnd`/`v`/`net_short`**：`derive_pins` 恒返回
2 个点、调用点不按 type 过滤，而 GT 的 `gnd` 元件恰好 1 脚 ⇒ 每元件白送 1 个 FP，strict 下
0 tp。见 MODEL_HANDOFF 76 + `integrate/probes/README.md`。

## 为什么现在还没有接进 runner

`runner.py` 里 box 元件**没有任何脚名来源**（`box_pinnum.md`：脚名不印在脚端点旁，
≤20px 只有 11.9%；真走"读脚名 + 指派"是另一个活）。没有来源就接 = 每次都返回空 = 死代码。
接线的位置已经定好：`runner.py` 主链路 `derive_pins` 循环**之后**，对"仍有 0 脚的 box"调用
`pins_from_names(c.name, named_points, ...)`。

## 基线（验收用，来自 ㉗ / `runs/stage05/probe_iclib*.md`）

| 来源 | 全脚命中（200 例 / 10gt） |
|---|---|
| 几何 `pin_1..N` 升序 | 21.59% / 21.59% |
| 官方库·**成员判据** | **53.5% / 97.9%** |
| **组合链**（本模块） | **48.3% / 90.5%** |

⚠ 这两个数是 **oracle 摆位 + oracle 脚名**，只换脚号来源 ⇒ 是**上界**，不是端到端可达值。
⚠ 精确的测量口径在 `probe_iclib.measure()`（已产物化，别再写第二遍）。
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from pcbmodel.integrate.numbering import DraftPin, pin_number_acceptable

# 官方 IC 库（唯一名判据）。**服务器上的绝对路径是默认值，行为与改动前逐字一致**；
# `PCB_IC_LIB` 只用来给容器换一个可读路径 —— 提交包里没有 `10gt/`，容器里那个绝对路径不存在，
# 缺了它 box 引脚链会 FileNotFoundError（Codex 审核 R1）。**只改路径来源，不改算法。**
LIB_PATH = Path(os.environ.get("PCB_IC_LIB")
                or Path(__file__).resolve().parents[2] / "10gt/10gt/BoxICLibrary_200cases.json")


# --------------------------------------------------------------------------- #
# 官方库
# --------------------------------------------------------------------------- #
def load_official(path: Path = LIB_PATH) -> dict:
    """`{型号: {脚名: [号, ...]}}`，按 `(脚名,号)` 去重（库里同脚会按 pin_type 重复列）。

    ⚠ 与 `probe_iclib.load_lib` 同实现。**故意重复 5 行**：那份是冻结证据（改它会作废
    `probe_iclib*.md` 的产物），这份是运行时。别让运行时依赖 probe。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    lib = {}
    for name, v in raw.items():
        by_name: dict[str, set] = defaultdict(set)
        for p in v.get("pins") or []:
            pn, num = str(p.get("name") or "").strip(), str(p.get("number") or "").strip()
            if pn and num:
                by_name[pn].add(num)
        lib[str(name).strip()] = {k: sorted(s) for k, s in by_name.items()}
    return lib


def package_n_pins(L: dict | None) -> int | None:
    """库里该型号的最大数字脚号 = **真实封装脚数**。

    ⚠ 这条是 MODEL_HANDOFF ㉚ 的修正：`pin_number_acceptable(k, n_pins)` 的 `n_pins` 不能拿
    GT 观测脚数 —— MSP432 的 GT 只标 27 脚，真实封装 128 脚，拿 27 去判会把合法的 28..128
    全当计数器号。官方库给的正是正确的 `n_pins`。
    """
    if not L:
        return None
    mx = [int(n) for v in L.values() for n in v if n.isdigit()]
    return max(mx) if mx else None


def _vote(table: dict | None, model: str, pinname: str, *, min_agree: int = 2) -> int | None:
    """投票表多数票；**平票弃权**。

    ⚠ 这里**不做 LOO**。留一减票需要 `(模型,脚名)` 的完整 Counter，而聚合表只存了
    `[号, 同意票, 总票]` —— 拿它减票**只在"表里多数恰好 == GT"时才对**（首版就这么错了）。
    LOO 是**测量**关切、运行时没有自票要减 ⇒ 由 `--report` 自己建 Counter 后喂一张临时表，
    本函数保持无状态。
    """
    row = ((table or {}).get("entries") or {}).get(str(model).strip())
    if not row:
        return None
    v = row.get(str(pinname).strip())
    if not v:
        return None
    k, agree, tot = int(v[0]), int(v[1]), int(v[2])
    if agree < min_agree or agree / max(tot, 1) < 0.6:
        return None
    return k


# --------------------------------------------------------------------------- #
# 核心：链
# --------------------------------------------------------------------------- #
def number_for(model: str | None, pinname: str, *, lib: dict, table: dict | None = None,
               n_pins: int | None = None, min_agree: int = 2) -> tuple[int | None, str]:
    """`(脚号, 来源)`；来源 ∈ {`lib`, `vote`, `abstain`}。

    链：**库唯一名 → 投票表 → 弃权**。库给多个号（VDD/GND 这类天然一对多）时**不猜**，
    落到投票表；两处都没有就返回 `None` —— **调用方不要写这个脚**（宁缺勿假）。
    """
    if not model:
        return None, "abstain"
    L = lib.get(str(model).strip())
    nums = (L or {}).get(str(pinname).strip())
    if nums and len(nums) == 1:
        k = int(nums[0]) if nums[0].isdigit() else None
        if k is None:
            return None, "abstain"
        if n_pins is None or pin_number_acceptable(str(k), n_pins):
            return k, "lib"
        return None, "abstain"
    k = _vote(table, model, pinname, min_agree=min_agree)
    if k is not None and (n_pins is None or pin_number_acceptable(str(k), n_pins)):
        return k, "vote"
    return None, "abstain"


def pins_from_names(model: str | None, named_points, *, lib: dict, table: dict | None = None,
                    n_pins: int | None = None,
                    min_agree: int = 2) -> tuple[list[DraftPin], Counter]:
    """`named_points` = 可迭代的 `(pinname, (x, y))`（已定位的脚名）。

    返回 `(pins, 来源计数)`。**查不到号的脚不进结果** —— 少写一个脚只是漏，
    写错一个脚既扣 tp 又添 fp。
    """
    by_model = (lib or {}).get(str(model).strip()) if model else None
    if n_pins is None:
        n_pins = package_n_pins(by_model)
    pins, C = [], Counter()
    for pinname, pt in named_points:
        k, src = number_for(model, pinname, lib=lib, table=table, n_pins=n_pins,
                            min_agree=min_agree)
        C[src] += 1
        if k is None:
            continue
        pins.append(DraftPin(point=(float(pt[0]), float(pt[1])),
                             pin_number_label=str(k), pinname=str(pinname)))
    return pins, C


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def selftest() -> None:
    lib = {
        "UNAMB": {"VCC": ["1"], "OUT": ["3"]},          # 唯一名
        "POWER": {"VDD": ["7", "16", "26"], "GND": ["2", "4"]},   # 一对多 ⇒ 不许猜
        "BIG": {"B": ["101"]},                           # 101 合法（封装 128 脚）；"A" 故意不在库里
    }
    table = {"entries": {"UNAMB": {"VCC": [1, 3, 3]}, "POWER": {"VDD": [7, 3, 3]},
                         "BIG": {"A": [9, 3, 3]}}}
    gp = lambda m, p: number_for(m, p, lib=lib, table=table, n_pins=None)

    assert gp("UNAMB", "VCC") == (1, "lib"), gp("UNAMB", "VCC")      # 库优先，压过投票表的 9
    assert gp("POWER", "VDD") == (7, "vote"), gp("POWER", "VDD")     # 一对多 ⇒ 落投票表
    assert gp("POWER", "GND") == (None, "abstain"), gp("POWER", "GND")   # 库里多号且表里没有
    assert gp("BIG", "A") == (9, "vote"), gp("BIG", "A")             # 库里没有 ⇒ 投票表
    assert gp("NOPE", "X") == (None, "abstain")
    assert gp(None, "X") == (None, "abstain")

    # n_pins 用库的封装脚数：BIG 封装 101 脚 ⇒ 101 合法；拿 GT 观测的 2 脚去判就会被误杀
    assert package_n_pins(lib["BIG"]) == 101
    assert package_n_pins(None) is None
    assert number_for("BIG", "B", lib=lib, n_pins=2) == (None, "abstain")   # 2 脚封装不会有 101
    assert number_for("BIG", "B", lib=lib, n_pins=None) == (101, "lib")

    # LOO：表里那一票减掉后只剩 2 票，仍够 min_agree=2；减到 1 票就弃权
    assert _vote(table, "UNAMB", "VCC") == 1        # 3/3
    assert _vote({"entries": {"M": {"P": [5, 2, 2]}}}, "M", "P") == 5
    assert _vote({"entries": {"M": {"P": [5, 1, 1]}}}, "M", "P") is None   # 只 1 票 < min_agree
    assert _vote({"entries": {"M": {"P": [5, 2, 4]}}}, "M", "P") is None   # 2/4=0.5 < 0.6

    # 组装：弃权的脚不写
    pins, c = pins_from_names("POWER", [("VDD", (1.0, 2.0)), ("GND", (3.0, 4.0))],
                              lib=lib, table=table)
    assert len(pins) == 1 and pins[0].pin_number_label == "7" and pins[0].pinname == "VDD"
    assert pins[0].point == (1.0, 2.0)
    assert c["vote"] == 1 and c["abstain"] == 1
    assert pins_from_names("NOPE", [("X", (0, 0))], lib=lib)[0] == []

    # 真 GT 冒烟：拿一个真 case 的 GT `(脚名, 点)` 喂进去，输出必须能被 scorer 消费
    tgt = _first_gt()
    if tgt is not None:
        pins, c = pins_from_names("U1", [("GPI04", (540.0, 854.0))], lib={}, table=None)
        assert pins == [] and c["abstain"] == 1        # 空库 ⇒ 一律弃权，不编号
        for desig, pd in list((tgt.get("pins") or {}).items())[:3]:
            named = [(v.get("pinname"), (v["point"]["x"], v["point"]["y"]))
                     for v in pd.values() if v.get("pinname")]
            got, _ = pins_from_names((tgt["components"][desig] or {}).get("Name"),
                                     named, lib={}, table=None)
            assert got == [], desig                    # 空库下恒空 —— 证明"不猜"是硬约定
    print("selftest OK (链三态 / 一对多不猜 / n_pins 用封装 / 平票弃权 / 弃权不写脚 / GT 冒烟)")


# --------------------------------------------------------------------------- #
# 跳 1 验收：喂 GT 脚名，只换脚号来源
# --------------------------------------------------------------------------- #
def report(min_agree: int = 2) -> dict:
    """**跳 1**：全 200 例 box 脚，喂 GT 脚名 + GT 摆位，只换脚号来源。

    对标 ㉗ 的组合链 **48.3%（全脚）/ 90.5%（10gt）**。
    ⚠⚠ **跳 1 通过 ≠ 能加分**：这里脚名与摆位都是抄答案的，端到端还要打折
    （脚名读取 + 指派 + 5px 定位），预期 **5–20%（待测）**。见 HANDOFF 79。
    """
    from pcbmodel.integrate import ic_db
    recs = ic_db.collect()
    lib = load_official()
    # 完整 Counter 才能做精确 LOO（聚合表做不到，见 `_vote` 注释）
    tab: dict = defaultdict(lambda: defaultdict(Counter))
    for r in recs:
        for pn, k in r["pins"].items():
            tab[r["name"]][pn][k] += 1

    C: Counter = Counter()
    for r in recs:
        if r["type"] != "box":
            continue
        L = lib.get(r["name"])
        n_pins = package_n_pins(L)
        for pn, key in r["pins"].items():
            C["pin"] += 1
            rem = Counter(tab[r["name"]][pn])
            rem[key] -= 1                                  # 减掉自己那一票
            tmp = {"entries": {}}
            if rem:
                top = rem.most_common()
                tmp["entries"] = {r["name"]: {pn: [int(top[0][0]), top[0][1], sum(rem.values())]}}
            k, src = number_for(r["name"], pn, lib=lib, table=tmp, n_pins=n_pins,
                                min_agree=min_agree)
            C[src] += 1
            if L is not None:
                C["inlib_pin"] += 1
            if k is not None:
                C["cov"] += 1
                C["hit"] += int(k == key)
                if L is not None:
                    C["inlib_cov"] += 1
                    C["inlib_hit"] += int(k == key)
    pct = lambda a, b: round(100.0 * a / max(b, 1), 1)
    C["hit_all_pct"] = pct(C["hit"], C["pin"])
    C["cov_pct"] = pct(C["cov"], C["pin"])
    C["hit_cov_pct"] = pct(C["hit"], C["cov"])
    C["hit_inlib_pct"] = pct(C["inlib_hit"], C["inlib_pin"])
    print("跳 1（喂 GT 脚名 + GT 摆位，只换脚号来源）｜ 官方 train 200 例")
    print(f"  box 脚 {C['pin']} ｜ 弃权 {C['abstain']}（库唯一名 {C['lib']} / 投票表 {C['vote']}）")
    print(f"  **全部 box 脚命中 {C['hit_all_pct']}%**（覆盖 {C['cov_pct']}%，覆盖内 {C['hit_cov_pct']}%）")
    print(f"  **型号在库里的脚 {C['inlib_pin']} 个命中 {C['hit_inlib_pct']}%**"
          f"  ← 与 ㉗ 的 48.3% **同口径**（㉗ 的分母只算在库型号）")
    print(f"  其余 {C['pin'] - C['inlib_pin']} 个脚属于库里没有的型号 ⇒ 只能靠投票表")
    print("  基线（㉗，同口径）：几何 pin_1..N 升序 21.59% ｜ 官方库成员判据 53.5% ｜ 组合链 48.3%")
    return dict(C)


def _first_gt():
    """找一个官方 train target 做冒烟；找不到就返回 None（CI 无数据集时不算失败）。"""
    root = Path("/home/linqs/code/CV for PCB/赛题六公开数据集/200_train_cases")
    if not root.is_dir():
        return None
    for d in sorted(root.iterdir()):
        for p in sorted(d.glob("*_target.json")):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report()
    elif "--selftest" in sys.argv or len(sys.argv) == 1:
        selftest()
    else:
        print(__doc__)
