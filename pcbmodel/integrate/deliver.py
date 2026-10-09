# -*- coding: utf-8 -*-
"""官方交付入口：`/input/*.png` → `/output/{decision_log.json, final_results/<case>_Pred.json}`。

契约（逐字照抄官方基础镜像 `/submission/README.md`，2026-09-15 到手）：

| | |
|---|---|
| 入口 | `/submission/agent_system -input /input -output /output` |
| 输入 | `/input/<case_name>.png` —— **一个目录多张图**、**不挂 GT**、只读 |
| 输出 | `/output/decision_log.json` + `/output/final_results/<case_name>_Pred.json`（每图一份） |
| 命名 | `case_name` = 图片文件名去扩展名，**含空格**（如 `0025-Altium Designer`） |
| 可选 | `/output/final_results/run.log`、`/output/runs/` |

配置 = 项目冻结的最强端到端（同 `report44.BEST`）：**`ocr-real`（detect 框）+ `--nets wire`**。

⚠ **device = 「先试 GPU，不可用则回落 CPU」**：官方镜像是 torch 2.5.1+cu121，**跑不了
sm_120（Blackwell）**，而 `torch.cuda.is_available()` 在那种卡上**仍返回 True** ⇒ 只看它
会整批静默 0 分。`pick_device` 因此**真起一个 kernel**。见 MODEL_HANDOFF「09-15 夜·三」。

本地跑（**pcbmodel env**）：
    python -m pcbmodel.integrate.deliver -input /tmp/in -output /tmp/out
"""
from __future__ import annotations

import os

# ⚠ 必须在 import ultralytics **之前**设：官方评测容器**可能没有外网**，而 ultralytics 会在
# 缺包时**自己 pip install**（实测：喂一张坏图就触发 `pi-heif` 下载，卡 7.6s、日志里刷一屏
# AutoUpdate）。我们要的是「要么跑要么干净报错」，不是「跑着跑着自己改环境」。
# 见 MODEL_HANDOFF「09-15 夜·三」⑥。
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.caseio import IMG_EXTS
from pcbmodel.integrate.runner import run_case
from pcbmodel.integrate.type_vocab import canonicalize_submission_file
from pcbmodel.integrate import name_fill as _namefill       # 【隔离候选】型号补填开关
from pcbmodel.ocr import match_13 as _match13            # 【隔离候选】开关读取

PROJ = Path(__file__).resolve().parents[2]
# 交付配置；**改这里就等于改交付**。与 report44.BEST 必须一致（`tests` 会比对）
# 2026-09-23 加 `box_pins="wire"`：box（IC）引脚链 —— val19 25.37→26.68（+1.31）、
# golden 35.09→36.64（+1.55），comp 逐位不变（HANDOFF 91）。
BEST = {"predictor": "ocr-real", "boxes": "detect", "nets": "wire", "box_pins": "wire"}
WEIGHTS = PROJ / "pcbmodel" / "runs" / "detect" / "train" / "yolo11m_v2" / "weights" / "best.pt"


def pick_device(pref: str = "0") -> tuple[str, str]:
    """先试 GPU，不可用则回落 CPU。返回 `(device, 理由)`。

    ⚠ **必须真算一下**，不能只看 `torch.cuda.is_available()`：sm_120 上它是 True，
    但 kernel 起不来 —— 实测失败形态很脏（检测预测失败 → 下游 `KeyError: 'all'`），
    **整批静默 0 分**。这里先比架构串（给出人话理由）再起一个 1 元素 kernel（兜住
    驱动/OOM 等一切其余情况）。
    """
    if pref == "cpu":
        return "cpu", "命令行显式指定"
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", "torch.cuda.is_available() = False"
        i = 0 if pref == "auto" else int(pref)
        name = torch.cuda.get_device_name(i)
        arch = "sm_" + "".join(str(v) for v in torch.cuda.get_device_capability(i))
        supported = torch.cuda.get_arch_list()
        if arch not in supported:
            return "cpu", f"{name} 是 {arch}，本 torch 只编了 {' '.join(supported)}"
        (torch.zeros(1, device=f"cuda:{i}") + 1).item()      # 真起一个 kernel
        return str(i), f"{name} ({arch})"
    except Exception as e:                                   # noqa: BLE001
        return "cpu", f"{type(e).__name__}: {e}"


def find_images(input_dir: Path) -> list[Path]:
    return sorted((p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS),
                  key=lambda p: p.name)


def _reclass_on() -> bool:
    """【隔离候选】薄封装，便于测试替换与阅读。"""
    return _match13.desig_reclass_on()


def _name_on() -> bool:
    """【隔离候选】IC 型号补填开关。"""
    return _namefill.name_fill_on()


def run_batch(input_dir: Path, output_dir: Path, device: str,
              limit: int = 0) -> tuple[list[dict], list[str]]:
    """跑一批。逐例**自己兜异常**（`run_case` 崩了也会留合法空壳）——
    官方是整批一次调用，一例挂掉不能带走其余的。"""
    from pcbmodel.data import loader
    loader.IMAGE_DIR = input_dir
    final = output_dir / "final_results"
    final.mkdir(parents=True, exist_ok=True)

    imgs = find_images(input_dir)
    if limit:
        imgs = imgs[:limit]
    recs: list[dict] = []
    log: list[str] = []
    promos: dict[str, dict] = {}          # 【隔离候选】逐例 {executed, items} 提升记录
    fills: dict[str, dict] = {}           # 【隔离候选】逐例 {executed, items} 型号补填记录
    for p in imgs:
        case = p.stem                                        # ← 官方就是「文件名去扩展名」
        if _reclass_on():
            # 每例**先清空**：`run_case` 在 OCR/检测早期就失败时不会写记录，不清就会把
            # 上一例的提升记录抄给这一例（审核 P2-2）。`None` 即"本例未执行到校正"。
            _match13.LAST_RECLASS = None
        if _name_on():
            _namefill.LAST_FILL = None       # 同上：失败例不得沿用上一例的补填记录
        run, _ = run_case(case, source="image", out_dir=final, score=False,
                          weights=str(WEIGHTS), device=device, **BEST)
        if _reclass_on():
            _rec = _match13.LAST_RECLASS
            # executed=False ⇒ 本例没跑到校正（区分「未执行」与「执行了但零提升」）
            promos[case] = {"executed": _rec is not None, "items": list(_rec or [])}
        if _name_on():
            _fill = _namefill.LAST_FILL
            fills[case] = {"executed": _fill is not None, "items": list(_fill or [])}
        # `run_case` 落的是 <case>.json（<case> 可含空格）；官方要 <case>_Pred.json，平铺
        src = final / f"{case}.json"
        if src.is_file():
            canonicalize_submission_file(src)  # 09-30 official type_list; after pin geometry
            src.rename(final / f"{case}_Pred.json")
        rec = {**run.to_dict(), "secs": round(run.secs, 2), "status":
               "ok" if (run.ok and not run.exc) else "error"}
        recs.append(rec)
        line = (f"[deliver] {case} {run.secs:6.2f}s comp={run.n_comp:4d} pin={run.n_pin:4d} "
                f"net={run.n_net:4d} {'ok' if rec['status'] == 'ok' else '❌ ' + str(run.exc)}")
        print(line, flush=True)
        log.append(line)
    if promos:                                               # 【隔离候选】开关关着时**不落文件**
        (output_dir / "desig_reclass.json").write_text(
            json.dumps({"env": _match13.DESIG_RECLASS_ENV,
                        "enabled": True, "promotions": promos},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    if fills:                                                # 同上：型号补填开关关着时不落
        (output_dir / "name_fill.json").write_text(
            json.dumps({"env": _namefill.NAME_FILL_ENV,
                        "enabled": True, "promotions": fills},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    return recs, log


def decision_log(input_dir: Path, output_dir: Path, device: str, why: str,
                 recs: list[dict], wall: float, cfg: dict | None = None) -> dict:
    """⚠ 官方说字段以《EDARace 手册》为准，而**该手册至今没到手** ⇒ 这里给**超集**：
    手册下来后按它裁字段即可，不用重跑。"""
    ok = sum(1 for r in recs if r["status"] == "ok")
    out = {
        "schema": "edarace-decision-log/1",
        "note": "字段以官方《EDARace 手册》为准；该手册未到手，本文件为超集",
        # ↓ 2026-10-08 平台适配：官方提交样例（2606sample_submission.zip）新增的四个字段。
        #   本管线是**纯 CV**，不调任何 API ⇒ token 一律如实记 0，不是「没写」。
        "method": "pure-cv",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processed_cases": len(recs),
        "token_usage": {"total_tokens": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "api_calls": 0},
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pipeline": dict(cfg or BEST),
        "device": {"used": device, "reason": why},
        "n_cases": len(recs), "n_ok": ok, "n_error": len(recs) - ok,
        "wall_s": round(wall, 1),
        "cases": recs,
    }
    # 【隔离候选】只在开关**打开**时加这一个自述字段 ⇒ 关着时交付件与生产同构。
    if _reclass_on():
        out["desig_reclass"] = {"env": _match13.DESIG_RECLASS_ENV, "enabled": True}
    if _name_on():
        out["name_fill"] = {"env": _namefill.NAME_FILL_ENV, "enabled": True}
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="agent_system",
                                 description="EDARace 交付入口（PNG 目录 → 官方输出契约）")
    ap.add_argument("-input", "--input", required=True, help="只读输入目录（多张 PNG）")
    ap.add_argument("-output", "--output", required=True, help="可写输出目录")
    ap.add_argument("--device", default="0", help='GPU 序号；"cpu" 强制 CPU；"auto" 取 0')
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 例（本地冒烟用）")
    a = ap.parse_args(argv)

    input_dir, output_dir = Path(a.input), Path(a.output)
    # 问题5修复：input 必须存在；output 不存在自动建（官方挂载大概率已建，但代价是整批 0 分）
    if not input_dir.is_dir():
        print(f"[deliver] input 目录不存在: {input_dir}", file=sys.stderr)
        return 3
    output_dir.mkdir(parents=True, exist_ok=True)
    device, why = pick_device(a.device)
    print(f"[deliver] device={device}（{why}）| 配置 {BEST}", flush=True)

    t0 = time.perf_counter()
    try:
        recs, log = run_batch(input_dir, output_dir, device, a.limit)
    except Exception as e:                                   # noqa: BLE001
        recs, log = [], [f"[deliver] 整批失败: {type(e).__name__}: {e}"]
        print(log[0], file=sys.stderr)
        traceback.print_exc()
    wall = time.perf_counter() - t0

    # 即使整批炸了也要落 decision_log.json / run.log —— 输出契约是硬要求
    log_text = json.dumps(decision_log(input_dir, output_dir, device, why, recs, wall),
                          ensure_ascii=False, indent=1)
    (output_dir / "decision_log.json").write_text(log_text, encoding="utf-8")
    (output_dir / "final_results").mkdir(parents=True, exist_ok=True)
    # ⚠ 位置冲突（官方资料自相矛盾，二选一都会赌）：基座 placeholder 与 `纯CV/agent.py` 都写
    #   `/output/decision_log.json`，只有样例总 README 写 `/output/final_results/decision_log.json`。
    #   这里**两处写同一份**：多出的那个不叫 `<case>_Pred.json`，而 `final_results/run.log`
    #   本来就是官方允许的同目录旁挂文件 ⇒ 被当成预测件评分的风险很低。平台实测后按结论裁掉一份。
    (output_dir / "final_results" / "decision_log.json").write_text(log_text, encoding="utf-8")
    (output_dir / "final_results" / "run.log").write_text(
        "\n".join(log) + "\n", encoding="utf-8")
    ok = sum(1 for r in recs if r["status"] == "ok")
    print(f"[deliver] {len(recs)} 例 ｜ ok {ok} ｜ error {len(recs) - ok} ｜ "
          f"device={device} ｜ {wall:.1f}s -> {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
