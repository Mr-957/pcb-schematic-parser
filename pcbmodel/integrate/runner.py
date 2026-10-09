# -*- coding: utf-8 -*-
"""P1 阶段 0.5.4：端到端 runner —— PNG → 合法 target JSON（210 例零崩溃）。

**流水线顺序（不可换）**：`草案 → 编号 → 净化 → 打分`。
净化器**绝不重新编号**，它只做格式修复；两者顺序颠倒会让"编号 bug"被净化器掩盖。

**四档 predictor**（决定"框"和"位号"从哪来）：

| predictor     | 框来源 | 位号来源 | 用途 |
|---|---|---|---|
| `empty`       | 无     | —        | 只验证管道不崩（分数必为 0） |
| `gt-boxes`    | **GT 框** | **自动编号** | 量"检测完美、但 OCR 位号没有"时的上限 |
| `oracle-keys` | **GT 框** | **GT 位号** | 量"检测与编号都完美"时的上限 |
| `ocr-sim`     | GT 框 | 只有图上真印着的位号"读得到" | 量「位号缺失」的代价 |
| `detect`      | **YOLO** | 自动编号（无 OCR） | **真实管线**（需 `--weights`，pcbmodel env） |
| `ocr-real`    | GT 或 YOLO | **真实 OCR**（1.1e 检测 + 1.2 识别 + 1.3 匹配） | 真端到端（需权重） |

⚠ **`--nets` 默认已是 `singleton`（每脚自成一网，2026-09-14 提默认）** —— 它不需要 GT net、
也不需要导线分割，实测真端到端 2.65→9.98。旧默认是 `none`（net 块恒 0）。
四档按"用了多少真信息"排：`none` < `singleton` < **`wire`**（真导线分割，带 `edges`，
**唯一能让 `line` 块出分**的档）< `gt`（抄答案，仅上限参照）。
**引用任何历史分数前先确认它的 nets 口径**（见 §八 风险 16）。

⚠ `gt-boxes` / `oracle-keys` **只允许在 train/golden 上报数**，任何情况下**不得**当 val 成绩 ——
它们用了 GT 框，是**上限参照**不是预测。

⚠ **`pinname` 非空**：`scorer.pin_score` 要求 `pinname` 与 pin key **同时**正确，
实测置空会让 pinF1 **从 0.46 掉到 0**。本 runner 对缺失的 `pinname` 用脚号回填并计入告警。

运行（pcbeval env，项目根）：
  python -m pcbmodel.integrate.runner --split all --predictor gt-boxes --out runs/stage05/pred
  python -m pcbmodel.integrate.runner --cases 0025 --predictor oracle-keys
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

from pcbeval.caseio import discover_official
from pcbeval.config import EvalConfig, OFFICIAL_GT_ROOT, OFFICIAL_TRAIN_ROOT
from pcbeval.refs import VOID
from pcbeval.scorer import evaluate as strict_evaluate
from pcbmodel.integrate import name_fill             # 【隔离候选】IC 型号补填开关
from pcbmodel.integrate import numbering as N
from pcbmodel.integrate import pin_geom
from pcbmodel.integrate.purify import purify

PROJ = Path(__file__).resolve().parents[2]
SPLIT_PATH = PROJ / "pcbmodel" / "data" / "train_val_split.json"
OUT_DIR = PROJ / "pcbmodel" / "runs" / "stage05"
# `--nets wire` 的默认分割权重（阶段 2；`unet_v1` 是 val19 下游加权分最优的那个）
SEG_CKPT = PROJ / "pcbmodel" / "runs" / "seg" / "unet_v1" / "best.pt"
PREDICTORS = ("empty", "gt-boxes", "oracle-keys", "ocr-sim", "detect", "ocr-real")


def _recoverable_upper(case_id: str, cfg: EvalConfig | None = None) -> set[str]:
    """该例**图上真的印着**的位号（去空白大写）—— 模拟"OCR 能读到的部分"。

    复用 `designator_availability` 的判据（已分类源文本 exact + merged 首 token）。
    """
    try:
        from pcbmodel.data.designator_availability import _texts_of_case
        got = _texts_of_case(case_id, cfg)
    except Exception:                                        # noqa: BLE001
        return set()
    if got is None:
        return set()
    _target, contents = got
    up = {c.strip().upper() for c in contents}
    up |= {c.split()[0].upper() for c in contents if c.split()}
    return up


@dataclass
class CaseRun:
    case_id: str
    ok: bool = False
    n_comp: int = 0
    n_pin: int = 0
    n_net: int = 0
    secs: float = 0.0
    n_fixed: int = 0
    n_dropped: int = 0
    n_warn: int = 0
    fatal: list[str] = field(default_factory=list)
    exc: str | None = None

    def to_dict(self) -> dict:
        return {"case": self.case_id, "ok": self.ok, "n_comp": self.n_comp,
                "n_pin": self.n_pin, "n_net": self.n_net,
                "repair_rate": (self.n_fixed + self.n_dropped),
                "n_fixed": self.n_fixed, "n_dropped": self.n_dropped,
                "n_warn": self.n_warn, "fatal": self.fatal[:5], "exc": self.exc}


# --------------------------------------------------------------------------- #
# 草案构造
# --------------------------------------------------------------------------- #
def singleton_nets(comps: list[N.DraftComp]) -> list[N.DraftNet]:
    """「每脚自成一网」（§八 风险 16）—— 一个引脚一个 net。

    这是**唯一一个既不偷 GT net、也不需要任何导线分割**就能构造 net 的策略：
    token 由 `numbering.apply` 从**管线自己**的位号/pin 键拼（`f"{desig}.{pin_key}"`），
    所以它对 `detect` / `ocr-real` / `gt-boxes` **三档都适用**（这正是它值钱的地方）。

    ⚠ `edges` **必须留空**：`line_score` 只比对**已配对** net 内的线段，把 GT edges 塞进来
    就能白拿 lineF1（实测 `nets="gt"` 在 oracle-keys 上 line=1.0000 —— 纯抄答案）。
    """
    return [N.DraftNet(members=[(i, j)], edges=[])
            for i, c in enumerate(comps) for j in range(len(c.pins))]


def draft_from_gt(case_id: str, *, source: str = "official-train",
                  with_pin_numbers: bool = True, with_pinname: bool = True,
                  with_desig_text: bool = False, nets: str = "none") -> N.DraftCase:
    """从官方 GT 造草案。

    `with_desig_text=True` 模拟"OCR 读到了位号"（oracle-keys）；
    默认 False → 由 `numbering` 自动编号（gt-boxes 上限）。

    `nets="gt"` 时用 GT 的 `hyperGraph` 回填 net 成员（**net 的完美上限参照**）；
    `nets="singleton"` = **每脚自成一网**（2026-09-12 新增，见 §八 风险 16）——
    这是**唯一一个既不偷 GT net 信息、也不需要任何导线分割**就能构造的 net 策略：
    token 由管线**自己**的位号/pin 键拼（`numbering.apply` 里 `f"{desig}.{pin_key}"`），
    故在 `gt-boxes` 档上等价于 GT token（c=1.0），量的就是这个策略的**真实增益**。
    ⚠ `edges` **必须留空**：给 GT edges 等于偷答案（`line_score` 只比对**已配对** net 内的线段）。
    `"none"` = net 全空 —— 那比"每脚自成一网"**更差**（netF1 恒 0）。
    ⚠ **本函数的 `nets` 默认值无所谓**（所有调用方都显式传）；**真正的默认在 CLI / `run_case`**，
    2026-09-14 起是 `singleton`。
    """
    from pcbeval.refs import TokenResolver, iter_tokens
    from pcbmodel.data.loader import load_sample
    s = load_sample(case_id, source=source, with_mask=False)

    comps: list[N.DraftComp] = []
    gt_desigs: list[str] = []
    gt_pin_keys: list[list[str]] = []
    for i, b in enumerate(s.comp_boxes):
        gt_desigs.append(b.desig)
        pins: list[N.DraftPin] = []
        keys: list[str] = []
        for p in s.pin_items:
            if p.desig != b.desig:
                continue
            keys.append(p.key)
            nlabel = None
            if with_pin_numbers:
                nlabel = p.key[4:] if p.key.startswith("pin_") else p.key
            pins.append(N.DraftPin(point=(p.point_up[0], p.point_up[1]),
                                   pin_number_label=nlabel,
                                   pinname=p.pinname if with_pinname else None))
        gt_pin_keys.append(keys)
        comps.append(N.DraftComp(bbox=list(b.bbox_up), ctype=b.type,
                                 name=b.name, value=b.value,
                                 desig_text=b.desig if with_desig_text else None,
                                 pins=pins))

    # GT 位号/pin 键 → 草案下标（草案按 GT 顺序构造，故位置即对应）
    dmap = {d: i for i, d in enumerate(gt_desigs)}
    resolver = TokenResolver(list(dmap),
                             {d: {k: {} for k in gt_pin_keys[i]}
                              for i, d in enumerate(gt_desigs)})
    nets_out: list[N.DraftNet] = []
    for _nm, net in (s.gt_nets or {}).items():
        edges = [[{"x": float(pt["x"]), "y": float(pt["y"])} for pt in seg]
                 for seg in (net.get("edges") or {}).values()
                 if isinstance(seg, (list, tuple)) and len(seg) == 2]
        members: list[tuple[int, int]] = []
        if nets == "gt":
            for tk in iter_tokens(net.get("hyperGraph", "")):
                got = resolver.resolve(tk)
                if got is None:
                    continue
                ci = dmap.get(got[0])
                if ci is None:
                    continue
                try:
                    pi = gt_pin_keys[ci].index(got[1])
                except ValueError:
                    continue
                members.append((ci, pi))
        nets_out.append(N.DraftNet(members=members, edges=edges))
    if nets == "singleton":
        # 丢掉上面那圈 GT net，改成每脚一网、**不给 edges**。
        # （上面那圈白跑了 —— 几百个 net，代价可忽略；换的是不用给主循环加缩进。ponytail）
        nets_out = singleton_nets(comps)
    return N.DraftCase(case_id=case_id, W=s.W, H=s.H, comps=comps, nets=nets_out,
                       gt_desigs=gt_desigs, gt_pin_keys=gt_pin_keys)


def draft_empty(case_id: str, source: str = "official-train") -> N.DraftCase:
    from pcbmodel.data.loader import load_sample
    s = load_sample(case_id, source=source, with_mask=False)
    return N.DraftCase(case_id=case_id, W=s.W, H=s.H, comps=[], nets=[])


@lru_cache(maxsize=2)
def _yolo(weights: str):
    """进程内复用 YOLO 实例。`draft_from_detect` 是按例调用的，每次重 load 一遍权重
    （~2s × N 例）纯属浪费。惰性导入 —— `runner.py` 的其余部分仍可在 pcbeval env 跑。"""
    from ultralytics import YOLO
    return YOLO(weights)


def draft_from_detect(case_id: str, *, source: str = "official-train",
                      weights: str | None = None, conf: float = 0.25,
                      iou: float = 0.5, device: str = "0",
                      plan=None, vocab_path: str | None = None,
                      nets: str = "none") -> N.DraftCase:
    """**真实检测**：YOLO 切块推理 → DraftComp（引脚由 `pin_geom` 几何推导）。

    ⚠ 必须在 **pcbmodel env** 跑（要 torch + ultralytics）；本函数对它们做**惰性导入**，
    这样 `runner.py` 的其余部分仍可在 pcbeval env 里用。

    ⚠⚠ **帧**：`detect/` 与 `tiles.py` 是 **y-down**，而 `pin_geom.derive_2pin` 只吃 **y-up**。
    这里显式转换，见 `eval_box._to_up_box` 与 MODEL_HANDOFF §八 风险 13。

    位号来源：本轮**没有 OCR**，所以 `desig_text` 一律 None → 由 `numbering` 自动编号。
    即这条路径等价于 `gt-boxes` 的"检测缺陷版"：框是预测的，位号仍是猜的。
    """
    from pcbmodel.data.loader import load_sample
    from pcbmodel.detect.eval_box import predict_case
    from pcbmodel.detect.tiles import DEFAULT_STRIDE, DEFAULT_TILE, TilePlan

    if not weights or not Path(weights).is_file():
        raise FileNotFoundError(f"detect 需要 --weights（找不到 {weights!r}）")
    plan = plan or TilePlan(DEFAULT_TILE, DEFAULT_STRIDE)
    # 词表由 dataset.py 生成在**数据集目录**（不是训练 run 目录）
    vp = Path(vocab_path) if vocab_path else (
        PROJ / "pcbmodel" / "runs" / "detect" / "ds_v2" / "vocab.json")
    if not vp.is_file():
        raise FileNotFoundError(f"找不到词表 {vp}（应由 dataset.py 生成）")
    with open(vp, encoding="utf-8") as f:
        vinfo = json.load(f)
    idx2name = {v: k for k, v in vinfo["vocab"].items()}

    model = _yolo(str(weights))
    s = load_sample(case_id, source=source, with_mask=False)
    dets = predict_case(s.image_orig, model, plan, conf=conf, iou=iou, device=device)

    comps: list[N.DraftComp] = []
    for det in dets:
        ty = idx2name.get(det.get("cls"), "other")
        bx_down = det["box"]
        bx_up = [bx_down[0], s.H - bx_down[3], bx_down[2], s.H - bx_down[1]]
        pins: list[N.DraftPin] = []
        got = pin_geom.derive_pins(bx_up, ty, 2)          # ← 必须 y-up
        if got:
            pins = [N.DraftPin(point=p, pin_number_label=str(i + 1))
                    for i, p in enumerate(got)]
        comps.append(N.DraftComp(bbox=bx_up, ctype=ty, pins=pins))
    return N.DraftCase(case_id=case_id, W=s.W, H=s.H, comps=comps,
                       nets=singleton_nets(comps) if nets == "singleton" else [])


@lru_cache(maxsize=1)
def _ocr_engine(det_weights: str | None, recog_ckpt: str | None,
                device: str, conf: float, iou: float):
    """OCR 引擎（1.1e 检测 + 1.2 识别）进程内单例。惰性导入 —— 其余档不受影响。"""
    from pcbmodel.ocr.predict_ocr import OcrEngine
    return OcrEngine(det_weights=det_weights, recog_ckpt=recog_ckpt,
                     device=device, det_conf=conf, det_iou=iou)


LAST_TEXT_STATS: dict = {}       # 诊断：最近一次 `draft_from_ocr` 的文本过滤计数


def drop_short_text_other_f(texts, maxlen: int = 2):
    """丢掉 `cls=="text_other"` 且去空白后长度 ≤ `maxlen` 的文本框。

    ⚠ **默认关闭**（`drop_short_text_other=False` ⇒ 逐位不变）。规则来自人工审核
    `low_conf` 855 条：在该池内杀 73% 的误检、误伤 2.7% 的"低置信但正确"。
    ⚠ **池内比例 ≠ 全量误伤率**（池子只含低置信条目）⇒ 晋升前必须看全量检出集上的测量
    与同进程 A/B（见 `probes/short_text_filter.py`）。
    """
    keep, dropped = [], 0
    for t in texts:
        if getattr(t, "cls", None) == "text_other" and len(str(t.content).strip()) <= maxlen:
            dropped += 1
            continue
        keep.append(t)
    return keep, dropped


def draft_from_ocr(case_id: str, *, source: str = "official-train",
                   boxes: str = "gt", weights: str | None = None,
                   det_weights: str | None = None, recog_ckpt: str | None = None,
                   conf: float = 0.25, iou: float = 0.5,
                   device: str = "0", plan=None, vocab_path: str | None = None,
                   nets: str = "none", drop_short_text_other: bool = False,
                   ) -> N.DraftCase:
    """**真实 OCR 管线**：框（GT 或 detect）+ 1.1e 文本检测 + 1.2 识别 + 1.3 匹配。

    `boxes="gt"`：框与引脚来自 GT（隔离检测缺陷），位号/脚号/引脚名/value 全来自真实 OCR
      —— 对比 `ocr-sim`（完美 OCR）即得真实 OCR 损失。
    `boxes="detect"`：框来自 YOLO（真端到端，需 `--weights`）—— 对比 `detect` 档
      （无 OCR）即得 OCR 增益。

    匹配用 content 键 + 几何 tiebreak（`pcbmodel/ocr/match_13.py`），不依赖 GT 位号。
    `nets`：`"none"` 或 `"singleton"`（每脚一网 —— 它对这档**也适用**，因为只需要管线自己的
    位号/pin 键）。`"gt"` 不行：该档位号是 OCR 预测的，回填不了 GT net 成员。
    """
    engine = _ocr_engine(det_weights, recog_ckpt, device, conf, iou)
    texts = engine.predict_case(case_id, source=source)
    n0 = len(texts)
    if drop_short_text_other:
        texts, _dropped = drop_short_text_other_f(texts)
    LAST_TEXT_STATS.update(n_text=n0, n_kept=len(texts), n_dropped=n0 - len(texts))

    if boxes == "gt":
        d = draft_from_gt(case_id, source=source, with_desig_text=False, nets="none")
    else:
        d = draft_from_detect(case_id, source=source, weights=weights, conf=conf,
                              iou=iou, device=device, plan=plan,
                              vocab_path=vocab_path)
    from pcbmodel.ocr import match_13
    # 【隔离候选】显式开关，默认关 ⇒ 下面一行不执行，本函数与生产逐字节等价。
    # 打开时在 `match()` **之前**就地把明确位号从 text_other 改成 designator；
    # 改的是同一个 `texts` 对象，因此下游 `d.box_texts`（box 引脚链）同源。
    if match_13.desig_reclass_on():
        match_13.LAST_RECLASS = match_13.reclassify_text_other(texts, d.comps)
    res = match_13.match(d.comps, texts)
    match_13.apply_to_draft(d, res)
    # box 引脚链的输入：**全部类**文本（不许按 `cls=='pinName'` 过滤，见 `box_pins` 模块头）。
    # 挂在 draft 上而不是塞进每一层签名：`ensure_pins` 在 `build_draft` 里晚一步才跑。
    d.box_texts = texts
    d.nets = singleton_nets(d.comps) if nets == "singleton" else []
    return d


@lru_cache(maxsize=1)
def _seg_model(ckpt: str, device: str):
    """U-Net 分割权重进程内单例（惰性导入 torch —— 本文件其余部分仍可在 pcbeval env 跑）。"""
    from pcbmodel.seg.unet import load_ckpt
    return load_ckpt(ckpt, device=device)


@lru_cache(maxsize=16)
def _case_segments(case_id: str, source: str, seg_ckpt: str | None, device: str,
                   thr: float):
    """U-Net 掩码 → 骨架 2 点段（y-up，与 `wire/ceiling` 同帧）。

    **按例缓存**：`--nets wire` 与 `--pin-anchor wire` 都要这一段几何，缓存避免同一例
    跑两遍 U-Net（单例滑窗是这条链上最贵的一步）。
    """
    from pcbmodel.data.loader import load_sample
    from pcbmodel.seg.unet import predict_mask
    from pcbmodel.wire.ceiling import mask_to_segments
    ck = Path(seg_ckpt) if seg_ckpt else SEG_CKPT
    if not ck.is_file():
        raise FileNotFoundError(f"需要分割权重，但找不到 {str(ck)!r}"
                                f"（--seg-weights 或先训 unet_v1）")
    s = load_sample(case_id, source=source, with_mask=False)
    dev = f"cuda:{device}" if str(device).isdigit() else device   # torch 要 "cuda:1"
    mask = predict_mask(_seg_model(str(ck), dev), s.image_up, device=dev, thr=thr)
    return tuple(mask_to_segments(mask))


def ensure_pins(draft: N.DraftCase, *, symbols: str = "off", wire_pts=None,
                tol: float | None = None, side_source: str = "off",
                img=None, box_pins: str = "off", box_texts=None,
                box_kw: dict | None = None, box_model_r: float | None = None,
                box_fuzzy1: bool = False, box_model_reuse: bool = False,
                stats: dict | None = None) -> int:
    """补引脚。**必须在 nets 构造之前调用** —— `--nets wire` 的 `wire_nets` 要看得见这些点。

    `symbols` 三档（默认 `off` = 历史行为逐位不变）：
      - `off`  ：只做 2 脚几何推导（现状）；
      - `prior`：再给 gnd/v 补 1 脚锚点（多数朝向，不看导线）；
      - `wire` ：用 `wire_pts` 做 wire-facing 选边；最近的候选都超出 `tol` 就不补。

    `box_pins` 两档（默认 `off` = 逐位不变，**HANDOFF 89/90 之后新加**）：
      - `off`：不碰多脚器件；
      - `wire`：给**仍然 0 脚的 `box`** 走 `integrate/box_pins.py` 的三段链
        （导线附着点 → 轴向邻域里的脚名文本 → 官方库/投票表换脚号），**查不到号就不写**。

    ⚠⚠ 1 脚符号的 comp key 是 ∅ 计数器 ⇒ **strict 下补点必然净负**（原来只算 1 个 FN，
    补完后 `fp += 1` 且 `fn += 1`，net 里还多一个对不上的 token）。收益只存在于
    "∅ 按坐标配对 + token 跟着重挂"的口径里（见 MODEL_HANDOFF 与账本的"口径赌注"条目）。
    返回补出的 1 脚锚点数（诊断用）。
    """
    n_sym = 0
    txs = None
    model_by_i: dict[int, str] = {}
    if box_pins != "off":
        from pcbmodel.integrate import box_pins as _bp
        # ⚠ 只在 box 链开着时才读 `draft.box_texts` —— 单测用的假 draft 没有这个字段
        txs = draft.box_texts if box_texts is None else box_texts
        # 真实管线里 box 的 `name`（型号串）**没人填**（2026-09-23 实测 32/32 弃权）
        # ⇒ 先从图上认型号（库里精确命中 + 型号形态 + 全局 1:1 最近优先），抢不到就弃权。
        todo = [i for i, c in enumerate(draft.comps)
                if c.ctype == "box" and not c.pins and not c.name]
        if todo:
            kw_m = {} if box_model_r is None else {"r": float(box_model_r)}
            local_models = _bp.resolve_models([draft.comps[i].bbox for i in todo],
                                               txs, fuzzy1=box_fuzzy1,
                                               reuse=box_model_reuse, **kw_m)
            # Resolver indices belong to the filtered box list, not draft.comps.
            model_by_i = {todo[j]: model for j, model in local_models.items()}
        if stats is not None:
            stats["box_model"] = stats.get("box_model", 0) + len(model_by_i)
            stats["box_comp"] = stats.get("box_comp", 0) + len(
                [1 for c in draft.comps if c.ctype == "box"])
    for i, c in enumerate(draft.comps):
        if box_pins != "off" and c.ctype == "box" and not c.pins:
            from pcbmodel.integrate import box_pins as _bp
            bp_stats = stats.setdefault("box_pins", {}) if stats is not None else None
            got = _bp.build(c.bbox, c.name or model_by_i.get(i), wire_pts, txs,
                            **(box_kw or {}), stats=bp_stats)
            if got:
                c.pins = got
                if stats is not None:
                    stats["box_pins_added"] = stats.get("box_pins_added", 0) + len(got)
            continue          # box 不进 2 脚推导、也不进 1 脚锚点（结构性不适用）
        if not c.pins:
            # pin_1 侧向：`side_source != off` 时用**图像**判（`pin_side`），经**已有的**
            # `pin1_side=` 注入 —— `pin_geom` 保持纯几何、一行不改（2026-09-22 定的分层）。
            side = None
            if side_source != "off" and img is not None:
                from pcbmodel.integrate import pin_side
                side = pin_side.side_of(img, c.bbox, c.ctype, source=side_source)
            got = pin_geom.derive_pins(c.bbox, c.ctype, 2, pin1_side=side)
            if got:
                c.pins = [N.DraftPin(point=p, pin_number_label=str(i + 1))
                          for i, p in enumerate(got)]
                continue
        if c.pins or symbols == "off":
            continue
        res = pin_geom.derive_symbol_anchor(c.bbox, c.ctype, wire_pts, tol=tol)
        if res is None:
            continue
        c.pins = [N.DraftPin(point=res[0], pin_number_label="1")]
        n_sym += 1
    return n_sym


def wire_nets(case_id: str, comps: list[N.DraftComp], *, source: str = "official-train",
              seg_ckpt: str | None = None, device: str = "0", thr: float = 0.5,
              tol: float | None = None, segs: list | None = None) -> list[N.DraftNet]:
    """**阶段 2 接进真管线**：U-Net 预测掩码 → 骨架段 → 按引脚点聚类 → net。

    与 `singleton_nets` 一样**不碰 GT net**，但多了**真的导线几何**（`edges`）⇒
    这是唯一能让 `line` 块（0.10 权重，至今恒 0）出分的档。三处口径：
      - **帧**：引脚点与掩码都在 **y-up**（`draft_from_detect` 已转、官方 labels 也是 y-up），
        与 `wire/ceiling.py` 同帧，**不做翻转**；
      - **身份用位置下标** `(comp_idx, pin_idx)`，**不用位号** —— 位号此时还没定
        （`numbering.apply` 之后才有），而 `DraftNet.members` 本来就是位置下标；
      - `edges` 是**预测**几何（不是 GT 抄来的），与 `--nets gt` 的抄答案有本质区别。
    ⚠ 必须在 **pcbmodel env**（torch + skimage）。
    """
    from pcbmodel.wire.ceiling import GT_TOL, cluster_components
    if segs is None:
        segs = _case_segments(case_id, source, seg_ckpt, str(device), float(thr))
    pins = {(i, j): (float(c.pins[j].point[0]), float(c.pins[j].point[1]))
            for i, c in enumerate(comps) for j in range(len(c.pins))}
    out: list[N.DraftNet] = []
    for keys, idxs in cluster_components(segs, pins, GT_TOL if tol is None else tol):
        out.append(N.DraftNet(members=sorted(keys), edges=[
            [{"x": a[0], "y": a[1]}, {"x": b[0], "y": b[1]}]
            for a, b in (segs[k] for k in idxs)]))
    return out


def build_draft(case_id: str, predictor: str, source: str, nets: str = "singleton",
                **kw) -> N.DraftCase:
    """`nets="wire"` = 先按 `none` 造草案，再整体换成**导线分割**出的 net（见 `wire_nets`）。

    走"先造后换"而不是往每个分支里塞参数：`wire` 只依赖 `comps`/`pins`，
    对**所有** predictor 都一样，塞进分支只会复制 4 遍（且要改 4 个函数签名）。
    """
    d = _draft_base(case_id, predictor, source,
                    "none" if nets == "wire" else nets, **kw)
    # ① 补引脚（含可选的 1 脚符号锚点）：**必须在 nets 之前**，且**两档走同一条** ——
    #    这样 singleton vs wire 的 comps/pins 仍逐位相同（ladder 的不变式不破）。
    symbols = kw.get("pin_anchor") or "off"
    side_source = kw.get("side_source") or "off"
    box_pins = kw.get("box_pins") or "off"
    segs = None
    # box 引脚链也要导线几何（脚点来自骨架段端点）⇒ 与 `wire` 两档共用按例缓存
    if symbols == "wire" or nets == "wire" or box_pins != "off":
        segs = _case_segments(case_id, source, kw.get("seg_weights"),
                              str(kw.get("device", "0")), float(kw.get("seg_thr", 0.5)))
    need_pts = symbols == "wire" or box_pins != "off"
    wire_pts = [p for a, b in segs for p in (a, b)] if (need_pts and segs) else None
    img = None
    if side_source != "off":
        from pcbmodel.data.loader import load_sample     # 只在需要时多载一次原图
        img = load_sample(case_id, source=source, with_mask=False).image_orig
    box_kw: dict = {}
    if kw.get("box_edge_tol") is not None:
        box_kw["edge_tol"] = float(kw["box_edge_tol"])
    if kw.get("box_r") is not None:
        box_kw["r"] = float(kw["box_r"])
    if kw.get("box_novote"):
        box_kw["use_vote"] = False          # 消融：只用官方库，关掉 GT 投票表
    ensure_pins(d, symbols=symbols, wire_pts=wire_pts,
                tol=kw.get("anchor_tol"), side_source=side_source, img=img,
                box_pins=box_pins, box_kw=box_kw, box_fuzzy1=bool(kw.get("box_fuzzy1", False)),
                box_model_reuse=bool(kw.get("box_model_reuse", False)),
                box_model_r=kw.get("box_model_r"), stats=kw.get("box_stats"))
    if nets == "wire":
        d.nets = wire_nets(case_id, d.comps, source=source, segs=segs,
                           seg_ckpt=kw.get("seg_weights"),
                           device=kw.get("device", "0"), thr=kw.get("seg_thr", 0.5))
    return d


def _draft_base(case_id: str, predictor: str, source: str, nets: str = "singleton",
                **kw) -> N.DraftCase:
    # kw 来自 CLI（weights/conf/device/vocab_path/boxes/det_weights/recog_ckpt），
    # 各分支**显式取参**——不能无脑 `**kw` 透传，否则多余参数会 TypeError。
    if predictor == "empty":
        return draft_empty(case_id, source)
    if predictor == "gt-boxes":
        return draft_from_gt(case_id, source=source, with_desig_text=False, nets=nets)
    if predictor == "oracle-keys":
        return draft_from_gt(case_id, source=source, with_desig_text=True, nets=nets)
    if predictor == "detect":
        # `--nets gt` 在 `detect` 档是**不可能**的（没有 OCR ⇒ 没有位号/pin 键 ⇒ 构造不出
        # GT net 成员），而它曾被**静默吞掉**产出 0 net。宁可报错也不静默（决策 22）。
        # `singleton` 可以：它只用管线自己的 comps/pins，不需要 GT net（§八 风险 16）。
        if nets not in ("none", "singleton"):
            raise ValueError(
                f"predictor={predictor!r} 不支持 --nets {nets!r}：该档无 GT 位号/pin 键，"
                f"构造不出 GT net 成员。要量 net 上限请用 "
                f"`--predictor gt-boxes --nets gt`（GT 框）；"
                f"`--nets singleton`（每脚一网）/ `--nets wire`（导线分割）是允许的。")
        # ponytail: 只拦 `detect`/`ocr-real`。`empty` 档 `--nets gt` 同样得 0 net，但那是它的
        # 定义（无 comps ⇒ 无 net 成员），没人会对它抱期望；而 `--matrix` 的 empty×gt 格子
        # 正因为不报错才可复现（`score.run_matrix`）。多拦一格要连带改矩阵，不划算。
        return draft_from_detect(case_id, source=source, weights=kw.get("weights"),
                                 conf=kw.get("conf", 0.25), iou=kw.get("iou", 0.5),
                                 device=kw.get("device", "0"), plan=kw.get("plan"),
                                 vocab_path=kw.get("vocab_path"), nets=nets)
    if predictor == "ocr-real":
        # 位号是 OCR 预测的 ⇒ 无 GT 位号/pin 键 ⇒ 构造不出 GT net 成员（同决策 22）。
        # `singleton` 同样允许（理由同 `detect`）。
        if nets not in ("none", "singleton"):
            raise ValueError(
                f"predictor={predictor!r} 不支持 --nets {nets!r}：该档位号是 OCR 预测的，"
                f"构造不出 GT net 成员（net 构建属阶段 2）；"
                f"`--nets singleton` / `--nets wire` 是允许的。")
        return draft_from_ocr(case_id, source=source,
                              drop_short_text_other=bool(kw.get("drop_short_text_other", False)),
                              boxes=kw.get("boxes", "gt"),
                              weights=kw.get("weights"),
                              det_weights=kw.get("det_weights"),
                              recog_ckpt=kw.get("recog_ckpt"),
                              conf=kw.get("conf", 0.25), iou=kw.get("iou", 0.5),
                              device=kw.get("device", "0"), plan=kw.get("plan"),
                              vocab_path=kw.get("vocab_path"), nets=nets)
    if predictor == "ocr-sim":
        # **模拟真实 OCR**：只有图上真的印着的位号才"读得到"，其余走自动编号。
        # 这是量「位号缺失」实际代价的口径 —— oracle-keys 会给答案，反而掩盖了这个损失。
        d = draft_from_gt(case_id, source=source, with_desig_text=False, nets=nets)
        rec = _recoverable_upper(case_id)
        for c, gd in zip(d.comps, d.gt_desigs):
            if gd and not gd.startswith(VOID) and gd.upper() in rec:
                c.desig_text = gd
        return d
    raise ValueError(f"未知 predictor {predictor!r}")


# --------------------------------------------------------------------------- #
# 单例
# --------------------------------------------------------------------------- #
def run_case(case_id: str, *, predictor: str = "gt-boxes",
             source: str = "official-train",
             policy: N.NumberingPolicy | None = None,
             out_dir: Path | None = None,
             cfg: EvalConfig | None = None,
             score: bool = True, nets: str = "singleton", keep_raw: bool = False,
             **kw) -> tuple[CaseRun, dict | None]:
    cfg = cfg or EvalConfig()
    t0 = time.time()
    run = CaseRun(case_id=case_id)
    pred = None
    try:
        draft = build_draft(case_id, predictor, source, nets=nets, **kw)
        # ① 补引脚：与 `build_draft` 同一个函数（幂等）。这里固定 `symbols="off"` —— 1 脚
        #    符号的锚点只能由 `--pin-anchor` 在 `build_draft` 里补，**必须早于 wire_nets**，
        #    放到这里就已经晚了（wire_nets 早跑完了）。
        ensure_pins(draft, symbols="off")
        # ② 编号（→ 合法 key）
        raw = N.apply(draft, policy)
        # ②′ 【隔离候选】IC 型号补填：**引脚与网络此时已定稿**（`N.apply` 的返回值就是最终
        #     JSON），这里只在空 Name 的 box 上补型号串 —— 不回流进 `draft`，因此碰不到
        #     pins/nets/key/bbox/type/value。开关默认关，关着时一行不执行（逐字节 == 生产）。
        if name_fill.name_fill_on():
            name_fill.apply_to_result(raw, draft, policy)
        # ③ 净化（**只做格式修复，绝不重新编号**）
        pred, rep = purify(raw, W=draft.W, H=draft.H, case_id=case_id)
        run.n_fixed = len(rep.fixed)
        run.n_dropped = len(rep.dropped)
        run.n_warn = len(rep.warnings)
        run.fatal = list(rep.fatal)
        run.ok = rep.ok
        run.n_comp = len(pred["components"])
        run.n_pin = sum(len(v) for v in pred["pins"].values())
        run.n_net = len(pred["nets"])
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{case_id}.json").write_text(
                json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
            # 净化**前**的产物：只在 --keep-raw 时落盘。默认不写 —— 每例多一个文件、
            # 200 例就翻倍占盘，而 `repair_rate` 是在内存里算的，不依赖它。
            # 真出现 repair_rate 飙升（= 上游模块坏了）时再开这个开关去 diff。
            if keep_raw:
                (out_dir / f"{case_id}.raw.json").write_text(
                    json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:                                   # noqa: BLE001
        run.ok = False
        run.exc = f"{type(e).__name__}: {e}"
        run.fatal.append(run.exc)
        if out_dir is not None:                              # 崩了也要留合法空壳，避免该例 0 分
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{case_id}.json").write_text(
                json.dumps({"components": {}, "pins": {}, "nets": {}},
                           ensure_ascii=False), encoding="utf-8")
    run.secs = time.time() - t0

    result = None
    if score and pred is not None:
        try:
            gt_path = discover_official(case_id, _root_for(source), kind=_kind(source)).target
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
            r = strict_evaluate(pred, gt, case_id=case_id, cfg=cfg)
            result = {"case": case_id, "score": r.score, "comp": r.comp["f1"],
                      "pin": r.pin["f1"], "net": r.net["f1"], "line": r.line["f1"]}
        except Exception as e:                               # noqa: BLE001
            run.fatal.append(f"score: {type(e).__name__}: {e}")
    return run, result


def _root_for(source: str):
    return OFFICIAL_GT_ROOT if source == "official-gt" else OFFICIAL_TRAIN_ROOT


def _kind(source: str) -> str:
    return "gt" if source == "official-gt" else "train"


def run_and_score_case(split: str = "golden", policy: N.NumberingPolicy | None = None,
                       predictor: str = "gt-boxes") -> list[dict]:
    """给 `numbering --sweep` 用：跑一个 split 并返回逐例分数。"""
    src = "official-gt" if split == "golden" else "official-train"
    out: list[dict] = []
    for cid in case_ids(split):
        _run, res = run_case(cid, predictor=predictor, source=src, policy=policy)
        out.append(res or {"case": cid, "score": None})
    return out


def case_ids(split: str = "all") -> list[str]:
    train = sorted(d.name for d in Path(OFFICIAL_TRAIN_ROOT).iterdir() if d.is_dir())
    golden = sorted(d.name.split("-", 1)[0] for d in Path(OFFICIAL_GT_ROOT).iterdir()
                    if d.is_dir())
    if split == "all":
        return train
    if split == "golden":
        return golden
    if SPLIT_PATH.is_file():
        sp = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        if split in sp:
            return list(sp[split])
        # `val_hard` 走上面这条通用分支即可（键就叫 val_hard）。
        # ⚠ 曾经的 `val_control` 特例已删：那个集合整批落在 train 里，是泄漏的评估集。
    raise ValueError(f"未知 split {split!r}")


def render_md(runs: list[CaseRun], agg: dict | None = None) -> str:
    ok = sum(1 for r in runs if r.ok)
    exc = sum(1 for r in runs if r.exc)
    L = ["# 端到端 runner 冒烟报告", "",
         "| 项 | 值 |", "|---|---:|",
         f"| 例数 | {len(runs)} |",
         f"| 净化后合法 | **{ok}/{len(runs)}** |",
         f"| 崩溃 | **{exc}** |",
         f"| 平均耗时 | {sum(r.secs for r in runs) / max(len(runs), 1):.2f}s |",
         f"| 有损丢弃总数 | {sum(r.n_dropped for r in runs)} |", ""]
    bad = [r for r in runs if r.exc or not r.ok]
    if bad:
        L += ["## 未通过", ""]
        for r in bad[:30]:
            L.append(f"- `{r.case_id}`: exc={r.exc} fatal={r.fatal[:2]}")
        L.append("")
    if agg:
        L += ["## 分数", "", "| 子集 | 例数 | 单例分 | compF1 | pinF1 | netF1 | lineF1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in agg.items():
            L.append(f"| {k} | {v['n']} | **{v.get('score', 0):.2f}** | "
                     f"{v.get('comp', 0):.4f} | {v.get('pin', 0):.4f} | "
                     f"{v.get('net', 0):.4f} | {v.get('line', 0):.4f} |")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="runner", description="端到端 runner（PNG → 合法 JSON）")
    ap.add_argument("--cases", default=None, help="逗号分隔 case id")
    ap.add_argument("--split", default=None,
                    choices=["all", "train", "val", "val_hard", "golden"])
    ap.add_argument("--predictor", default="gt-boxes", choices=list(PREDICTORS))
    ap.add_argument("--policy", default="default", choices=sorted(N.POLICIES))
    ap.add_argument("--out", default=str(OUT_DIR / "pred"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default=None,
                    choices=["official-train", "official-gt"],
                    help="覆盖 GT 来源（--cases 单跑时用；默认按 --split 推断）")
    ap.add_argument("--weights", default=None, help="detect/ocr-real(boxes=detect) 的元件 YOLO 权重")
    ap.add_argument("--vocab", default=None, help="类别词表（默认 ds_v2/vocab.json）")
    ap.add_argument("--boxes", default="gt", choices=["gt", "detect"],
                    help="ocr-real 的框来源：gt=GT 框（隔离检测缺陷）；detect=YOLO 框（真端到端）")
    ap.add_argument("--det-weights", default=None,
                    help="ocr-real 的 1.1e 文本检测权重（默认 text_v1/best.pt）")
    ap.add_argument("--recog-ckpt", default=None,
                    help="ocr-real 的 1.2 识别权重（默认 recog_v1_anno1694_seed6_500ep/best.pt，"
                         "端到端实测 14.91（`nets=none` 口径）/ 23.74（`singleton`）"
                         " 优于第 3 轮 11.60）")
    ap.add_argument("--keep-raw", action="store_true",
                    help="额外落盘净化前的 <id>.raw.json（调试 repair_rate 用；默认不写）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seg-weights", default=str(SEG_CKPT),
        help="`--nets wire` 的 U-Net 分割权重（阶段 2，默认 unet_v1/best.pt）")
    ap.add_argument("--seg-thr", type=float, default=0.5,
        help="`--nets wire` 的掩码阈值")
    ap.add_argument("--pin-anchor", default="off", choices=["off", "prior", "wire"],
        help="1 脚符号（gnd/v/net_*）连接点：off=不补（历史行为，默认）｜prior=按多数朝向补"
             "｜wire=用导线选边（wire-facing，四候选取最近）。⚠⚠ **strict 口径下补点净负**"
             "（∅ key 对不上：FN→FP+FN），收益只在 '∅ 按坐标配对 + token 重挂' 的口径里；"
             "验收必须 strict / loose / loose+token 三列并报（见 `ensure_pins`）")
    ap.add_argument("--side-source", default="off", choices=["off", "geo", "model", "parallel"],
        help="pin_1 侧向来源：off=常数表（默认，逐位不变）｜geo=极性标记｜model=图像分类器"
             "｜parallel=分类器为主+几何交叉校验。⚠ 常数表 100%% 的类（r/c/l/fuse/crystal）"
             "永不被覆盖；冲突/低置信自动回退常数表（见 `pin_side.side_of`）")
    ap.add_argument("--box-pins", default="off", choices=["off", "wire"],
        help="**box（多脚器件）引脚链**（HANDOFF 89/90）：off=不碰（默认，逐位不变）｜"
             "wire=给仍然 0 脚的 box 走「导线附着点 → 轴向邻域脚名文本 → 官方库/投票表换脚号」，"
             "查不到号就不写（宁缺勿假）。需 `--seg-weights`（共用 `--nets wire` 的按例缓存），"
             "且只有 `--predictor ocr-real` 有脚名文本 ⇒ 其余档自动不生效")
    ap.add_argument("--drop-short-text-other", action="store_true",
        help="丢掉 cls=='text_other' 且长度<=2 的文本框（**默认关闭**；规则见 probes/short_text_filter.py）")
    ap.add_argument("--box-model-reuse", action="store_true",
        help="box 型号：1:1 贪心后**只给没配到的 box 补空**，允许同一条型号文本服务多个元件"
             "（同型号常只印一次）；**已配上的元件一律不改**。默认关闭（逐位不变）")
    ap.add_argument("--box-fuzzy1", action="store_true",
        help="box 型号：精确未命中时仅接受库内编辑距离 <=1 的唯一合法型号；默认关闭")
    ap.add_argument("--box-edge-tol", type=float, default=None,
        help="box 候选脚点：导线端点距元件框 ≤ 该值（默认 20px；跳 0：框外中位 10 / p90 19）")
    ap.add_argument("--box-r", type=float, default=None,
        help="box 脚名候选：到「脚点→框内」射线段 ≤ 该值（默认 15px；跳 0：r10~r20）")
    ap.add_argument("--box-model-r", type=float, default=None,
        help="box 型号串搜索半径：文本中心到元件框 ≤ 该值（默认 100px；"
             "型号只能从图上认，见 `box_pins.resolve_models`）")
    ap.add_argument("--box-novote", action="store_true",
        help="消融：box 号源只用**官方 IC 库**，关掉由官方 train GT 汇总的投票表")
    ap.add_argument("--nets", default="singleton",
                    choices=["none", "gt", "singleton", "wire"],
                    help="**默认 singleton = 每脚自成一网**（不偷答案、不需分割，见 §八 风险 16；"
                         "实测真端到端 2.65→9.98）｜ none=空 net（旧默认，net 块恒 0）"
                         " ｜ wire=**导线分割**出的 net（阶段 2 接入，唯一能让 line 块出分的档，"
                         "需 `--seg-weights`，pcbmodel env）"
                         " ｜ gt=用 GT hyperGraph 回填成员（**偷答案，仅作上限参照，line 列不可引**）")
    a = ap.parse_args(argv)

    if a.cases:
        ids = [x.strip() for x in a.cases.split(",") if x.strip()]
    elif a.split:
        ids = case_ids(a.split)
    else:
        ap.print_help()
        return 0
    if a.limit:
        ids = ids[:a.limit]

    source = a.source or ("official-gt" if a.split == "golden" else "official-train")
    out_dir = Path(a.out)
    runs: list[CaseRun] = []
    box_stats: dict = {}
    for cid in ids:
        run, _res = run_case(cid, predictor=a.predictor, source=source,
                             policy=N.POLICIES[a.policy], out_dir=out_dir, nets=a.nets,
                             weights=a.weights, conf=a.conf, device=a.device,
                             vocab_path=a.vocab, keep_raw=a.keep_raw,
                             boxes=a.boxes, det_weights=a.det_weights,
                             recog_ckpt=a.recog_ckpt,
                             seg_weights=a.seg_weights, seg_thr=a.seg_thr,
                             pin_anchor=a.pin_anchor, side_source=a.side_source,
                             box_pins=a.box_pins, box_edge_tol=a.box_edge_tol,
                             box_r=a.box_r, box_model_r=a.box_model_r,
                             box_novote=a.box_novote, box_fuzzy1=a.box_fuzzy1,
                             drop_short_text_other=a.drop_short_text_other,
                             box_model_reuse=a.box_model_reuse,
                             box_stats=None if a.box_pins == "off" else box_stats)
        runs.append(run)
    ok = sum(1 for r in runs if r.ok)
    exc = sum(1 for r in runs if r.exc)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "runner_smoke.md").write_text(render_md(runs), encoding="utf-8")
    print(f"[runner] {len(runs)} 例 ｜ 合法 {ok} ｜ 崩溃 {exc} ｜ predictor={a.predictor}"
          f" ｜ policy={a.policy} -> {out_dir}")
    if a.box_pins != "off":
        print(f"[runner] box 链：型号认出 {box_stats.get('box_model', 0)}/"
              f"{box_stats.get('box_comp', 0)} 个元件 ｜ "
              f"补出 {box_stats.get('box_pins_added', 0)} 个脚 ｜ 明细 {box_stats.get('box_pins')}")
    for r in runs:
        if r.exc or not r.ok:
            print(f"   ⚠ {r.case_id}: {r.exc or r.fatal[:2]}")
    return 0 if (exc == 0 and ok == len(runs)) else 1


if __name__ == "__main__":
    sys.exit(main())
