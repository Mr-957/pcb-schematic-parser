# -*- coding: utf-8 -*-
"""官方 target 的 token 解析权威（hyperGraph ↔ designator/pin）与 ∅ 归一化。

**为什么单独成模块**：`hyperGraph` 的 token 形如 `designator.pin后缀`，而**位号可以含点**
（如 `U2.3` + `pin_2` → token `U2.3.2`），**pin 后缀也可以含点**（如 `U7` + `pin_0618.1`
→ token `U7.0618.1`）。因此**按点拆（任一方向）都会错**：
  - 从左往右拆：`U2.3.438` → 位号 `U2`（不存在）
  - 从右往左拆：`U7.0618.1` → 位号 `U7.0618`（不存在）
唯一正确解法 = **对 `components` 键集做最长前缀匹配**。实测 210 个官方文件 22059 个
token 全部可解析。

⚠ `赛题六公开数据集/HANDOFF.md` §11.5 建议「从右往左拆最后一个点」**是错的**，以本模块为准。

⚠ `∅` 的正确码点是 **U+2205 EMPTY SET**。qwen 兜底链路可能产出 U+00D8 `Ø`、U+2300 `⌀`、
U+03A6 `Φ` 等形近字——严格 key 口径下码点错 = 静默全不匹配。用 `normalize_void()` 归一。

坐标/帧无关：本模块只处理**字符串结构**，不碰坐标。

运行（pcbeval env，项目根）：
  python -m pcbeval.refs --audit        # 审计全部 210 例：token 可解析率 / 往返一致性
  python -m pcbeval.refs --case 0108    # 单例解析明细
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

from pcbeval.caseio import discover_official
from pcbeval.config import OFFICIAL_GT_ROOT, OFFICIAL_TRAIN_ROOT

# `∅` U+2205 的形近字 → 全部归一到 U+2205。
# 只在**首位**做归一（见 normalize_void），避免把 value 里合法的 `Ø10mm` 也改掉。
VOID = "∅"
VOID_LOOKALIKES = frozenset([
    "Ø",  # Ø LATIN CAPITAL LETTER O WITH STROKE
    "ø",  # ø LATIN SMALL LETTER O WITH STROKE
    "Φ",  # Φ GREEK CAPITAL LETTER PHI
    "ϕ",  # ϕ GREEK PHI SYMBOL
    "ɸ",  # ɸ LATIN SMALL LETTER PHI
    "⌀",  # ⌀ DIAMETER SIGN
    "Ф",  # Ф CYRILLIC CAPITAL LETTER EF
    "⨯",  # ⨯ (偶见 OCR 误识)
])

PIN_PREFIX = "pin_"


def iter_tokens(hyper: str) -> list[str]:
    """`hyperGraph` `"(tok,tok,...)"` → token 列表。

    口径与 `pcbeval.scorer._hyper_tokens` / `pcbeval.sanity._hyper_tokens` **完全一致**：
    **只按逗号切分，不拆 token 内部**。空串与 `"()"` 都返回 `[]`。

    ⚠ 不可再引入任何其它分隔（token 里真实存在空格、点、`/`、`+`、CJK、希腊字母，
    例如 `U2.2K6 3 29`、`U1.4.7μF`、`USB1.一24`）。
    """
    s = (hyper or "").strip()
    if not s:
        return []
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return [tk.strip() for tk in s.split(",") if tk.strip()]


def make_hypergraph(tokens) -> str:
    """token 序列 → 官方格式 `"(t1,t2,...)"`（**逗号后无空格**）。

    对任意官方 net 满足往返恒等：`make_hypergraph(iter_tokens(hg)) == hg`。
    """
    return "(" + ",".join(tokens) + ")"


def join_token(designator: str, pin_key: str) -> str:
    """`("U2.3", "pin_2")` → `"U2.3.2"`。pin_key 必须带 `pin_` 前缀。"""
    suf = pin_key[len(PIN_PREFIX):] if pin_key.startswith(PIN_PREFIX) else pin_key
    return f"{designator}.{suf}"


def normalize_void(s: str) -> str:
    """把**首位**的 `∅` 形近字归一成 U+2205，并做 NFC 规范化。

    只在首位归一：`Ø10mm` 这类出现在 value 里的合法文本不应被改动。
    """
    if not s:
        return s
    s = unicodedata.normalize("NFC", s)
    if s[0] in VOID_LOOKALIKES:
        return VOID + s[1:]
    return s


class TokenResolver:
    """按 `components` 键集做**最长前缀**匹配，把 token 还原成 `(位号, pin 键)`。

    每例构造一次（键集 ~45 个、token ~105 个，无需 trie）。解析结果带缓存。

    `resolve()` 的取舍：**优先取「位号是合法键 且 余部是该元件合法 pin 键」的最长前缀**；
    若没有任何前缀能给出合法 pin 键，退化为「最长的合法位号前缀」，让净化器有机会去
    补造该 pin —— 这比丢掉 token 划算（见 `pcbmodel/integrate/purify.py` 的取舍论证）。
    """

    __slots__ = ("_keys", "_pins", "_maxlen", "_cache")

    def __init__(self, comp_keys, pins: dict | None = None) -> None:
        self._keys = frozenset(comp_keys)
        self._pins = pins or {}
        self._maxlen = max((len(k) for k in self._keys), default=0)
        self._cache: dict[str, tuple[str, str] | None] = {}

    def resolve(self, token: str) -> tuple[str, str] | None:
        """token → `(designator, "pin_<后缀>")`；无法解析返回 None。"""
        hit = self._cache.get(token, False)
        if hit is not False:
            return hit  # type: ignore[return-value]
        fallback: tuple[str, str] | None = None
        res: tuple[str, str] | None = None
        # 前缀长度从长到短；n <= len-2 保证余部至少 1 字符
        lo = min(len(token) - 2, self._maxlen)
        for n in range(lo, 0, -1):
            if token[n] != ".":
                continue
            cand = token[:n]
            if cand not in self._keys:
                continue
            if self._pins:
                pk = PIN_PREFIX + token[n + 1:]
                if pk in (self._pins.get(cand) or {}):
                    res = (cand, pk)
                    break
                if fallback is None:
                    fallback = (cand, pk)   # 位号对但 pin 不存在
            else:
                res = (cand, PIN_PREFIX + token[n + 1:])
                break
        if res is None:
            res = fallback
        self._cache[token] = res
        return res

    def legal(self) -> set[str]:
        """该例全部合法的 `designator.pin后缀` 字符串（口径同 `sanity.py`）。"""
        return {f"{d}.{pk[len(PIN_PREFIX):]}"
                for d, pd in self._pins.items() for pk in pd}

    def __contains__(self, token: str) -> bool:
        r = self.resolve(token)
        return r is not None and r[1] in (self._pins.get(r[0]) or {})


def resolve_tokens(tokens, comp_keys, pins: dict | None = None
                   ) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """批量入口 → `(可解析映射, 无法解析列表)`。净化器要的就是这两个。"""
    r = TokenResolver(comp_keys, pins)
    ok: dict[str, tuple[str, str]] = {}
    bad: list[str] = []
    for tk in tokens:
        got = r.resolve(tk)
        if got is None:
            bad.append(tk)
        else:
            ok[tk] = got
    return ok, bad


def audit_case(case_id: str, target: dict) -> dict:
    """单例审计：token 可解析率、往返一致性、∅ 码点异常。"""
    comps = target.get("components", {})
    pins = target.get("pins", {})
    keys = list(comps)
    res = TokenResolver(keys, pins)
    n_tok = n_any = n_pin_ok = 0
    unresolved: list[str] = []
    pin_missing: list[str] = []
    roundtrip_bad: list[str] = []
    void_bad: list[str] = []
    for name, net in target.get("nets", {}).items():
        hg = net.get("hyperGraph", "")
        toks = iter_tokens(hg)
        if make_hypergraph(toks) != hg:
            roundtrip_bad.append(name)
        for tk in toks:
            n_tok += 1
            got = res.resolve(tk)
            if got is None:
                unresolved.append(tk)
                continue
            n_any += 1
            if got[1] in (pins.get(got[0]) or {}):
                n_pin_ok += 1
            else:
                pin_missing.append(tk)
    for k in list(comps) + list(pins) + list(target.get("nets", {})):
        if k and k[0] in VOID_LOOKALIKES:
            void_bad.append(k)
    return {"case_id": case_id, "n_tokens": n_tok, "n_resolved": n_any,
            "n_pin_exists": n_pin_ok, "unresolved": unresolved,
            "pin_missing": pin_missing, "roundtrip_bad": roundtrip_bad,
            "void_bad": void_bad}


def _official_targets():
    """(case_id, target, root, kind) —— 200_train 平铺 + 10_GT 带 tool 子目录。"""
    for root, kind in ((OFFICIAL_TRAIN_ROOT, "train"), (OFFICIAL_GT_ROOT, "gt")):
        if not Path(root).is_dir():
            continue
        for d in sorted(Path(root).iterdir()):
            if not d.is_dir():
                continue
            cid = d.name.split("-", 1)[0] if kind == "gt" else d.name
            try:
                oc = discover_official(cid, root, kind=kind)
            except Exception:
                continue
            if oc.target is None:
                continue
            try:
                with open(oc.target, encoding="utf-8") as f:
                    t = json.load(f)
            except Exception:
                continue
            yield cid, t


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="refs", description="官方 target 的 token 解析审计（最长前缀 + ∅ 归一）")
    ap.add_argument("--audit", action="store_true", help="审计全部 210 例")
    ap.add_argument("--case", default=None, help="单例明细")
    ap.add_argument("--show", type=int, default=0, help="明细打印条数上限")
    a = ap.parse_args(argv)

    if a.case:
        for cid, t in _official_targets():
            if cid != a.case:
                continue
            r = audit_case(cid, t)
            print(f"[refs] {cid}: tokens={r['n_tokens']} resolved={r['n_resolved']} "
                  f"pin_exists={r['n_pin_exists']} unresolved={len(r['unresolved'])} "
                  f"pin_missing={len(r['pin_missing'])}")
            for tk in r["unresolved"][:a.show or 10]:
                print(f"   UNRESOLVED  {tk!r}")
            for tk in r["pin_missing"][:a.show or 10]:
                print(f"   PIN-MISSING {tk!r}")
            return 0
        print(f"[refs] 找不到 case {a.case}")
        return 1

    tot = {"files": 0, "tokens": 0, "resolved": 0, "pin_exists": 0}
    bad_unres: list[tuple[str, str]] = []
    bad_pinmiss: list[tuple[str, str]] = []
    bad_rt: list[tuple[str, str]] = []
    bad_void: list[tuple[str, str]] = []
    n_nets = 0
    for cid, t in _official_targets():
        r = audit_case(cid, t)
        tot["files"] += 1
        tot["tokens"] += r["n_tokens"]
        tot["resolved"] += r["n_resolved"]
        tot["pin_exists"] += r["n_pin_exists"]
        n_nets += len(t.get("nets", {}))
        bad_unres += [(cid, x) for x in r["unresolved"]]
        bad_pinmiss += [(cid, x) for x in r["pin_missing"]]
        bad_rt += [(cid, x) for x in r["roundtrip_bad"]]
        bad_void += [(cid, x) for x in r["void_bad"]]

    print(f"[refs] 审计 {tot['files']} 例 / {n_nets} 个 net / {tot['tokens']} 个 token")
    print(f"  可解析（位号命中） : {tot['resolved']}/{tot['tokens']} = "
          f"{100 * tot['resolved'] / max(tot['tokens'], 1):.2f}%")
    print(f"  且 pin 键存在      : {tot['pin_exists']}/{tot['tokens']} = "
          f"{100 * tot['pin_exists'] / max(tot['tokens'], 1):.2f}%")
    print(f"  无法解析           : {len(bad_unres)}")
    print(f"  可解析但 pin 缺失  : {len(bad_pinmiss)}")
    print(f"  hyperGraph 往返不一致 : {len(bad_rt)}")
    print(f"  ∅ 形近字异常       : {len(bad_void)}")
    for tag, bad in (("UNRES", bad_unres), ("PINMISS", bad_pinmiss),
                     ("RT", bad_rt), ("VOID", bad_void)):
        for cid, x in bad[:10]:
            print(f"    {tag} {cid} {x!r}")
    ok = not (bad_unres or bad_rt or bad_void)
    print("[refs] AUDIT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
