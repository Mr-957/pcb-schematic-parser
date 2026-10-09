# -*- coding: utf-8 -*-
"""IC 引脚映射表：`(型号, 脚名) → 脚号`。**供 pinName 识别拿到后查脚号用。**

## 这张表补的是哪一环

`scorer.pin_score` 要求 **key 与 pinname 同时相等**，而`probe_box_pinnum` 已量出
「两样同时对」在最乐观假设下只有 **0.8%(≤20px)** —— **瓶颈是脚名读数，不是脚号**。
所以本表的定位是**第二半**：等脚名读出来之后，用它换到脚号。

⚠⚠ **别在没解决脚名之前指望它。** 同口径（oracle 摆位 + oracle 脚名，只换 key 来源）
对比 `probe_box_keys`：几何 `pin_1..pin_N` x 升序 = **0.2159**，本表（LOO）= **0.7263**
（box 件）。**3.4 倍，但全部要乘在脚名命中率上。**

## 表怎么来的

官方 200 例 train 的 GT，按 **cross-case 多数票**汇总。**这就是训练数据**（和权重同性质），
不是外部 datasheet。⚠ 交付前请确认赛题规则对「把训练集 GT 汇总成查表」是否有额外限制。

⚠⚠ **GT 有两处硬错误，表必须防住**：
  ① `type=='box'` 的脚里 **26% 的 key > 100**（最大一条 21 位垃圾数 `206161161612111119211`）
     —— 那是官方生成期的**全局计数器**（`∅` 那条 `pin_578` 同款），不是封装脚号。
     真实 datasheet 只会有 `1..48`，**再准的 DB 也对不上这 1/4**。
  ② 同一 `(型号, 脚名)` 在「见过 ≥2 次」的组里只有 **53.7%** 自洽。
⇒ 所以存**票数**、查表时要求**多数且票数够**，并复用 `numbering.pin_number_acceptable`
  拦掉计数器号。宁可弃权（返回 None）也不要吐一个假脚号。

## 键的选型（实测，别改回去）

`(型号, 脚名)` **优于** `(型号, 脚数, 脚名)`（box 件 LOO 命中 0.7263 vs 0.7084，
覆盖还更高）—— **脚数不是有效的消歧维度**：那些冲突是噪声（计数器逐 case 漂移），
不是「同型号不同封装」。见 `runs/stage05/ic_db.md`。

跑（pcbeval env 即可，无 torch）：
  <env>/bin/python -m pcbmodel.integrate.ic_db --build      # 生成 ic_pin_db.json
  <env>/bin/python -m pcbmodel.integrate.ic_db --report     # LOO 实测（诚实口径）
  <env>/bin/python -m pcbmodel.integrate.ic_db --selftest   # 无需磁盘的自检
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = Path(__file__).resolve().parent / "ic_pin_db.json"
_SUF = re.compile(r"^pin_(\d+)(?:_\d+)?$")     # GT 的重号后缀（`pin_8` / `pin_8_1`）

META = "__meta__"


# --------------------------------------------------------------------------- #
# 采集
# --------------------------------------------------------------------------- #
def collect(case_ids=None) -> list[dict]:
    """官方 train 的 GT -> `[{case, name, type, n_pins, pins: {脚名: 脚号}}]`。

    ⚠ 同名脚（GT 用 `pin_8_1` 表示 VDDA/VSSA 同号）会**互相覆盖**，这是刻意的：
    表是 `脚名 -> 脚号` 的多→一映射，冲突交给票数处理。
    """
    from pcbeval.caseio import discover_official
    from pcbeval.config import OFFICIAL_TRAIN_ROOT
    if case_ids is None:
        case_ids = sorted(d.name for d in OFFICIAL_TRAIN_ROOT.iterdir() if d.is_dir())
    out = []
    for cid in case_ids:
        t = json.loads(discover_official(cid, OFFICIAL_TRAIN_ROOT,
                                         kind="train").target.read_text(encoding="utf-8"))
        pins = t.get("pins", {})
        for des, c in t["components"].items():
            got: dict[str, int] = {}
            for k, v in pins.get(des, {}).items():
                m = _SUF.match(k)
                if not m:
                    continue
                pn = str(v.get("pinname") or "").strip()
                if pn:
                    got[pn] = int(m.group(1))
            if got:
                out.append({"case": cid, "name": str(c.get("Name") or "").strip(),
                            "type": c.get("type"), "n_pins": len(got), "pins": got})
    return out


# --------------------------------------------------------------------------- #
# 建表 / 存取 / 查表
# --------------------------------------------------------------------------- #
def build(recs: list[dict]) -> dict:
    """多数票汇总。值 = `[脚号, 同意票, 总票]`。"""
    tab: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in recs:
        if not r["name"]:
            continue
        for pn, k in r["pins"].items():
            tab[r["name"]][pn][k] += 1
    ent: dict[str, dict[str, list]] = {}
    for nm, bypn in sorted(tab.items()):
        row = {}
        for pn, c in sorted(bypn.items()):
            (k, agree), tot = c.most_common(1)[0], sum(c.values())
            row[pn] = [k, agree, tot]
        ent[nm] = row
    types = Counter(r["type"] for r in recs)
    return {META: {"n_comp": len(recs), "n_name": len(ent),
                   "cases": len({r["case"] for r in recs}),
                   "types": dict(types)},
            "entries": ent}


def save(table: dict, path: Path = DB_PATH) -> Path:
    path.write_text(json.dumps(table, ensure_ascii=False, indent=0, sort_keys=True),
                    encoding="utf-8")
    return path


def load(path: Path = DB_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lookup(table: dict, name: str, pinname: str, n_pins: int | None = None,
           *, min_agree: int = 2, min_ratio: float = 0.6) -> int | None:
    """查脚号。查不到/不够硬 ⇒ `None`（**宁缺勿假**）。

    默认 `min_agree=2`：只见一次的组合等于瞎猜（LOO 里那 6224 个组合绝大多数只出现一次）。
    `min_ratio` 挡掉「3 票里 2 票」这种勉强多数。⚠ 最后还要过
    `numbering.pin_number_acceptable`，**拦掉 GT 的计数器号**（48 脚件不该有 pin_183）。
    """
    row = (table.get("entries") or {}).get(str(name).strip())
    if not row:
        return None
    v = row.get(str(pinname).strip())
    if not v:
        return None
    k, agree, tot = int(v[0]), int(v[1]), int(v[2])
    if agree < min_agree or agree / max(tot, 1) < min_ratio:
        return None
    if n_pins is not None:
        from pcbmodel.integrate.numbering import pin_number_acceptable
        if not pin_number_acceptable(str(k), n_pins):
            return None
    return k


# --------------------------------------------------------------------------- #
# 诚实口径：LOO
# --------------------------------------------------------------------------- #
def evaluate_loo(recs: list[dict], *, min_agree: int = 2) -> dict:
    """留一实测：全表建一次，评估时把**该元件自己的那一票减掉**（O(N)）。

    ⚠ 这一层不能省。表是从这 200 例建的，**拿它评这 200 例 = 自问自答**。
    """
    tab: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in recs:
        for pn, k in r["pins"].items():
            tab[r["name"]][pn][k] += 1
    res = {"cov": 0, "hit": 0, "tie": 0, "n_pin": 0, "comp_full": 0, "comp": 0,
           "box_cov": 0, "box_hit": 0, "by_type": defaultdict(lambda: [0, 0])}
    for r in recs:
        res["comp"] += 1
        res["n_pin"] += len(r["pins"])
        g = 0
        for pn, k in r["pins"].items():
            rem = Counter(tab[r["name"]][pn])
            rem[k] -= 1                                  # 减掉自己那一票
            if sum(rem.values()) < min_agree:
                continue
            # ⚠⚠ **平票要留在 `cov` 里**：`lookup()` 对平票返回 None（min_ratio=0.6 挡住
            #   1:1 的 0.5）⇒ 消费方拿不到脚号 = **失手**，和"没查到"同价。
            #   把它从分母排掉会把命中率**虚高**（首版就这么错了：box 0.8526 vs 诚实 0.7263）。
            res["cov"] += 1
            g += 1
            t2 = res["by_type"][r["type"]]
            t2[0] += 1
            if r["type"] == "box":
                res["box_cov"] += 1
            top = rem.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                res["tie"] += 1
                continue
            ok = int(top[0][0] == k)
            res["hit"] += ok
            t2[1] += ok
            if r["type"] == "box":
                res["box_hit"] += ok
        if g == len(r["pins"]):
            res["comp_full"] += 1
    return res


def _render(recs: list[dict], r: dict) -> str:
    def bt(ty):
        c, h = r["by_type"][ty]
        return f"| `{ty}` | {c} | {h} | {h / max(c, 1):.4f} |"
    rows = "\n".join(bt(t) for t in sorted(r["by_type"], key=lambda t: -r["by_type"][t][0]))
    return "\n".join([
        "### IC 引脚映射表（`(型号, 脚名) → 脚号`）｜ LOO 实测", "",
        "| 量 | 数 |", "|---|---:|",
        f"| 元件 | {r['comp']} |", f"| 脚 | {r['n_pin']} |",
        f"| 有历史可查的脚（剩余票数 ≥2） | {r['cov']} / {r['n_pin']} = "
        f"**{r['cov'] / max(r['n_pin'], 1):.2%}** |",
        f"| ↳ 其中**平票**（`lookup` 返回 None，**已计入分母**） | {r['tie']} = "
        f"{r['tie'] / max(r['cov'], 1):.2%} |",
        f"| ↳ **脚号命中** | **{r['hit']}** |",
        f"| **命中率（平票算失手）** | **{r['hit'] / max(r['cov'], 1):.4f}** |",
        f"| ↳ ⚠ 若把平票从分母排掉（**虚高，别引用**） | "
        f"{r['hit'] / max(r['cov'] - r['tie'], 1):.4f} |",
        f"| **每一个脚都查得到的元件** | {r['comp_full']} / {r['comp']} = "
        f"{r['comp_full'] / max(r['comp'], 1):.2%} |", "",
        f"| ⭐ **`box`（IC，本表的目标）** | 可查 {r['box_cov']} 脚，"
        f"命中 **{r['box_hit'] / max(r['box_cov'], 1):.4f}**"
        f"（{r['box_hit']}/{r['box_cov']}） |", "",
        "**按 type 拆（口径同上：平票计入分母）**：", "",
        "| type | 可查脚 | 命中 | 命中率 |", "|---|---:|---:|---:|", rows, "",
        "**读法**：",
        "1. ⚠ **`box`（IC）才是本表的目标** —— 非 box 的脚名常**本身就是脚号**"
        "（`PH-4A` 的脚名就是 `2`/`4`），查得到但**没有增量**；那些 1.0000 是"
        "「脚名 == 脚号」的同义反复，**别当成模型的功劳**。",
        "2. ⚠ 命中率**必须与 `probe_box_keys` 的 0.2159 比**（同口径 oracle 摆位+脚名，"
        "只换 key 来源）—— 别看绝对值，看倍数。",
        "3. ⚠ 覆盖是**交付例与这 200 例的型号重合度**的函数，**我们无法预先知道** —— "
        "真实覆盖率只会 ≤ 这里的数（这里还是 LOO，交付要跨数据集）。", "",
    ])


# --------------------------------------------------------------------------- #
# 自检（不碰磁盘）
# --------------------------------------------------------------------------- #
def selftest() -> None:
    recs = [
        {"case": "a", "name": "ICX", "type": "box", "n_pins": 3,
         "pins": {"VCC": 1, "GND": 2, "OUT": 3}},
        {"case": "b", "name": "ICX", "type": "box", "n_pins": 3,
         "pins": {"VCC": 1, "GND": 2, "OUT": 3}},
        {"case": "c", "name": "ICX", "type": "box", "n_pins": 3,
         "pins": {"VCC": 1, "GND": 9, "OUT": 3}},          # GND 分歧 -> 多数仍是 2
        {"case": "d", "name": "CNT", "type": "box", "n_pins": 2,
         "pins": {"P": 183, "Q": 40000}},                  # 计数器号
        {"case": "e", "name": "ONE", "type": "box", "n_pins": 1, "pins": {"P": 7}},
    ]
    t = build(recs)
    assert lookup(t, "ICX", "VCC") == 1
    assert lookup(t, "ICX", "GND") == 2, lookup(t, "ICX", "GND")   # 2/3 多数
    assert lookup(t, "ICX", "NOPE") is None                        # 没这个脚名
    assert lookup(t, "NOSUCH", "VCC") is None                      # 没这个型号
    # 只出现一次 -> min_agree=2 挡住（不是"查到了"，是"没证据"）
    assert lookup(t, "ONE", "P") is None
    assert lookup(t, "ONE", "P", min_agree=1) == 7
    # ⚠ 计数器号：给 n_pins 就该被 pin_number_acceptable 拦掉
    assert lookup(t, "CNT", "Q", min_agree=1) == 40000             # 不给 n_pins = 不设防
    assert lookup(t, "CNT", "Q", 2, min_agree=1) is None, "48 脚件不该有 pin_40000"
    # 平票 -> 弃权
    t2 = build([{"case": "a", "name": "TIE", "type": "box", "n_pins": 2,
                 "pins": {"X": 1}}, {"case": "b", "name": "TIE", "type": "box",
                                     "n_pins": 2, "pins": {"X": 2}}])
    assert lookup(t2, "TIE", "X") is None, "1:1 平票必须弃权"
    # n_pins 缺省时不设防（调用方自己负责）
    assert lookup(t, "ICX", "VCC", 3) == 1
    print("selftest OK：多数票 / 票数不足 / 平票弃权 / 计数器号拦截 / 未知型号")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ic_db")
    ap.add_argument("--build", action="store_true", help=f"生成 {DB_PATH.name}")
    ap.add_argument("--report", action="store_true", help="LOO 实测")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(DB_PATH))
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    if a.selftest:
        selftest()
        return 0
    if not (a.build or a.report):
        ap.print_help()
        return 0

    from pcbeval.config import OFFICIAL_TRAIN_ROOT
    ids = sorted(d.name for d in OFFICIAL_TRAIN_ROOT.iterdir() if d.is_dir())
    recs = collect(ids)
    if a.build:
        p = save(build(recs), Path(a.out))
        n = len(load(p)["entries"])
        print(f"[ic_db] {p}  型号 {n} 个  元件 {len(recs)} 个  "
              f"{p.stat().st_size / 1024:.0f} KB")
    if a.report:
        from pathlib import Path as _P
        md = _P(__file__).resolve().parents[1] / "runs" / "stage05" / "ic_db_table.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        txt = _render(recs, evaluate_loo(recs))
        md.write_text(txt, encoding="utf-8")
        print(txt)
        print(f"[ic_db] -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
