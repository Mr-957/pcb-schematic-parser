# -*- coding: utf-8 -*-
"""P1 阶段 0.5.6：位号可得性标记（决策 19 第 6 件；源自 §零 归因分析 5）。

**要解决的问题**：命名元件的**位号本身常常不在图上**。官方 target 里 `U1`/`C7` 这样的
key 是从**原始 CAD 源**带过来的，而发布出来的 PNG 里有相当一部分根本没印位号
（`0003` 印的是 `U?`；`0028`/`0075` 整张图只有 value 与器件型号）。**这不是 OCR 没读出来，
是图上没有** —— 没有视觉模型能恢复它。

**为什么必须标记**：官方 component/pin 都是**严格 key** 口径，位号猜不中 ⇒ 该元件的
compF1 锁死为 0，且 hyperGraph token 形如 `designator.pin`，**netF1（权重 0.35）被连带污染**。
若不区分，全体分数会被这 28 个案例系统性拉低，**误判成"模型没学到"**。

**判据（三档，全部基于源 CVJsonStd `texts`）**
  - `exact`  ：存在一条文本内容 **去空白大写后 == 位号**
  - `merged` ：存在一条「位号+value 并成一个框」的文本（如 `R11 100`），其**首个 token == 位号**
  - `miss`   ：都没有 ⇒ 判为不可恢复

⚠ `exact + merged` 是**推理可恢复率的乐观上界**：源 `texts.content` 本身是有噪声的 OCR，
本模块只回答「图上有没有」，不回答「我们的 OCR 能不能读对」。
⚠ 与 `runs/stage0/text_geometry.md` Q9 的 82.8% **不是一回事** —— 那是**训练对**数量
（已知答案、只需位置），**不是**推理可恢复率。

运行（pcbeval env，项目根）：
  python pcbmodel/data/designator_availability.py                    # 全 200 例
  python pcbmodel/data/designator_availability.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.caseio import CaseError, discover_case_files, discover_official, load_cvjsonstd
from pcbeval.config import EvalConfig, OFFICIAL_GT_ROOT, OFFICIAL_TRAIN_ROOT
from pcbeval.gtbuilder import is_noise
from pcbmodel.data.loader import normalize_detail

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "stage0"
DEFAULT_THRESHOLD = 0.5
VOID = "∅"


@dataclass
class Availability:
    case_id: str
    n_named: int              # 命名元件数（不含 ∅）
    n_exact: int
    n_merged: int
    n_other: int              # 有文本但不是位号（value/pinName 之类）

    @property
    def n_recoverable(self) -> int:
        return self.n_exact + self.n_merged

    @property
    def ratio(self) -> float:
        return self.n_recoverable / self.n_named if self.n_named else 0.0

    def winnable(self, threshold: float = DEFAULT_THRESHOLD) -> bool:
        """位号可恢复率 ≥ threshold ⇒ 该例落在「可赢子集」。"""
        return self.n_named > 0 and self.ratio >= threshold

    def to_dict(self) -> dict:
        return {"n_named": self.n_named, "n_exact": self.n_exact,
                "n_merged": self.n_merged, "n_other": self.n_other,
                "n_recoverable": self.n_recoverable,
                "ratio": round(self.ratio, 4),
                "winnable": self.winnable()}


def _texts_of_case(case_id: str, cfg: EvalConfig | None = None) -> tuple[dict, list[str]] | None:
    """→ `(官方 target, 源文本内容列表)`；无 CVJsonStd 时返回 None。"""
    cfg = cfg or EvalConfig()
    oc = discover_official(case_id, OFFICIAL_TRAIN_ROOT, kind="train")
    with open(oc.target, encoding="utf-8") as f:
        target = json.load(f)
    try:
        cv = load_cvjsonstd(discover_case_files(case_id, cfg).cvjson)
    except CaseError:
        return None
    out: list[str] = []
    for t in cv.get("texts", {}).values():
        if normalize_detail(t.get("details")) is None:
            continue                      # 未分类（details=''）不进判据
        c = (t.get("content") or "").strip()
        if not c or is_noise(c):
            continue
        out.append(c)
    return target, out


def designator_availability(case_id: str, cfg: EvalConfig | None = None) -> Availability | None:
    """单例位号可得性。无 CVJsonStd 或该例无命名元件时返回 None。"""
    got = _texts_of_case(case_id, cfg)
    if got is None:
        return None
    target, contents = got
    named = [d for d in target.get("components", {}) if not d.startswith(VOID)]
    if not named:
        return None
    upper = {c.strip().upper() for c in contents}
    first = {c.split()[0].upper() for c in contents if c.split()}
    n_ex = n_mg = 0
    for d in named:
        du = d.upper()
        if du in upper:
            n_ex += 1
        elif du in first:
            n_mg += 1
    return Availability(case_id=case_id, n_named=len(named), n_exact=n_ex,
                        n_merged=n_mg, n_other=len(named) - n_ex - n_mg)


def train_case_ids(root: Path = OFFICIAL_TRAIN_ROOT) -> list[str]:
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir())


def golden_ids(root: Path = OFFICIAL_GT_ROOT) -> set[str]:
    return {d.name.split("-", 1)[0] for d in Path(root).iterdir() if d.is_dir()}


def build_table(case_ids=None, cfg: EvalConfig | None = None,
                threshold: float = DEFAULT_THRESHOLD) -> dict:
    """全量构建。返回可直接落盘的 dict（含 per-case + 子集清单 + meta）。"""
    cfg = cfg or EvalConfig()
    ids = list(case_ids) if case_ids is not None else train_case_ids()
    per: dict[str, dict] = {}
    skipped: list[str] = []
    for cid in ids:
        try:
            a = designator_availability(cid, cfg)
        except Exception:
            a = None
        if a is None:
            skipped.append(cid)
            continue
        per[cid] = a.to_dict()

    n_named = sum(v["n_named"] for v in per.values())
    n_rec = sum(v["n_recoverable"] for v in per.values())
    zero = sorted(k for k, v in per.items() if v["n_recoverable"] == 0)
    winnable = sorted(k for k, v in per.items() if v["winnable"])
    nonwinnable = sorted(k for k, v in per.items() if not v["winnable"])
    dist = Counter(v["n_recoverable"] == 0 for v in per.values())

    return {
        "threshold": threshold,
        "per_case": per,
        "zero_cases": zero,
        "winnable": winnable,
        "nonwinnable": nonwinnable,
        "skipped": sorted(skipped),
        "meta": {
            "source": "赛题六公开数据集/200_train_cases",
            "n_cases": len(per),
            "named_components": n_named,
            "recoverable": n_rec,
            "recoverable_rate": round(n_rec / n_named, 4) if n_named else 0.0,
            "n_zero_cases": len(zero),
            "n_zero_components": sum(per[k]["n_named"] for k in zero),
            "definition": "exact = 源文本去空白大写 == 位号；"
                          "merged = 源文本首 token == 位号（位号与 value 并框）；"
                          "判据只用**已分类**文本（details 非空且非噪声）",
            "caveat": "exact+merged 是**推理可恢复率的乐观上界**：源 texts.content 本身是"
                      "有噪声的 OCR，本表只回答「图上有没有」，不回答「读不读得对」。"
                      "⚠ 与 text_geometry.md Q9 的 82.8% 不是一回事（那是**训练对**数量）。",
            "usage": "打分时必须分开报『全体』与『可赢子集』，否则会被零位号案例系统性拉低",
        },
    }


def render_md(t: dict) -> str:
    m = t["meta"]
    L = ["# 位号可得性（阶段 0.5.6 / §零 归因分析 5）", "",
         f"> 源：`{m['source']}`（{m['n_cases']} 例）｜阈值 {t['threshold']}", "",
         "## 结论", "",
         "| 项 | 值 |", "|---|---:|",
         f"| 命名元件总数 | {m['named_components']} |",
         f"| 位号可从源文本找回 | {m['recoverable']}"
         f"（**{m['recoverable_rate'] * 100:.1f}%**） |",
         f"| 一个都找不回的案例 | **{m['n_zero_cases']}** |",
         f"| 这些案例覆盖的命名元件 | {m['n_zero_components']}"
         f"（占 {m['n_zero_components'] / max(m['named_components'], 1) * 100:.1f}%） |",
         f"| 可赢子集（ratio ≥ {t['threshold']}） | {len(t['winnable'])} 例 |",
         f"| 非可赢子集 | {len(t['nonwinnable'])} 例 |", "",
         "## 零位号案例（位号在图上一处都没有）", "",
         "```", " ".join(t["zero_cases"]) or "（无）", "```", "",
         "## 判据", "", f"- {m['definition']}", f"- ⚠ {m['caveat']}", "",
         "## 用法", "", f"- {m['usage']}", ""]
    return "\n".join(L)


def load_table(path: Path | None = None) -> dict:
    p = path or (OUT_DIR / "designator_availability.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def winnable_subset(case_ids=None, *, threshold: float = DEFAULT_THRESHOLD,
                    table: dict | None = None) -> set[str]:
    """给定 case 列表 → 其中「位号可赢」的子集。无表时现算。"""
    t = table or build_table()
    per = t["per_case"]
    ids = list(case_ids) if case_ids is not None else list(per)
    return {c for c in ids
            if c in per and per[c]["n_recoverable"] / max(per[c]["n_named"], 1) >= threshold}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="designator_availability",
                                 description="位号可得性标记（哪些案例的位号根本不在图上）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 例（0=全部）")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--check", action="store_true", help="断言已知结论（回归用）")
    a = ap.parse_args(argv)

    ids = train_case_ids()
    if a.limit:
        ids = ids[:a.limit]
    t = build_table(ids, threshold=a.threshold)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "designator_availability.json").write_text(
        json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "designator_availability.md").write_text(render_md(t), encoding="utf-8")
    m = t["meta"]
    print(f"[avail] {m['n_cases']} 例 ｜ 命名元件 {m['named_components']} ｜ "
          f"可找回 {m['recoverable']}（{m['recoverable_rate'] * 100:.1f}%）｜ "
          f"零位号案例 {m['n_zero_cases']} ｜ 可赢子集 {len(t['winnable'])} -> {out}")

    if a.check:
        errs = []
        if not (0.55 <= m["recoverable_rate"] <= 0.72):
            errs.append(f"可找回率 {m['recoverable_rate']} 偏离实测带 0.55~0.72")
        if not (20 <= m["n_zero_cases"] <= 40):
            errs.append(f"零位号案例数 {m['n_zero_cases']} 偏离实测带 20~40")
        if t["zero_cases"] and "0028" not in t["zero_cases"]:
            errs.append("0028 应判为零位号（已目视确认整张图无位号）")
        # 0025 的位号 R4~R7 已确认印在图上且被 OCR 读到（`texts` 里 details='designator'），
        # 必须判为可恢复。⚠ 别拿 0119 当锚点：它只有 1 个命名元件且恰好没找回，
        # 按判据**合法地**落在零位号集合里（样本太小，不具判别力）。
        if "0025" in t["per_case"] and t["per_case"]["0025"]["n_recoverable"] == 0:
            errs.append("0025 的位号 R4~R7 印在图上且被 OCR 读到，不应判为零位号")
        if "0025" in t["per_case"] and t["per_case"]["0025"]["n_exact"] == 0:
            errs.append("0025 至少应有 exact 级命中")
        if errs:
            print("[avail] CHECK FAIL:")
            for e in errs:
                print("  -", e)
            return 1
        print("[avail] CHECK PASS：可找回率与零位号案例数落在实测带内")
    return 0


if __name__ == "__main__":
    sys.exit(main())
