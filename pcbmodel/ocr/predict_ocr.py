# -*- coding: utf-8 -*-
"""端到端 OCR 引擎：1.1e 文本检测（text_v1）→ 1.2 识别（CRNN）→ OcrText。

管线（对单例）：
  load_sample → predict_case(1.1e, 切块 1280/768) → 每框裁剪 →
  _preprocess_crop（灰度/竖排旋转/resize 高 32，与 train_recog.load_img 逐行一致）
  → CRNN greedy_decode → (bbox y-up, cls, content, conf)。

只读依赖：`pcbmodel/runs/detect/train/text_v1/weights/best.pt`（1.1e）、
`pcbmodel/runs/ocr/ds_text_v1/vocab.json`（类别词表）、
`pcbmodel/runs/ocr/recog_v1_anno1694_seed6_500ep/best.pt`（1.2）。不写任何 detect/ 文件。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import os
import torch

PROJ = Path(__file__).resolve().parents[2]
TEXT_DET_DEFAULT = PROJ / "pcbmodel" / "runs" / "detect" / "train" / "text_v1" / "weights" / "best.pt"
TEXT_VOCAB_DEFAULT = PROJ / "pcbmodel" / "runs" / "ocr" / "ds_text_v1" / "vocab.json"
# 默认用 baseline（无竖排旋转）：端到端实测 14.91 vs 第 3 轮 11.60（val 19 例，同预处理），
# 第 4 轮（回滚旋转+丢弃竖排，v4）端到端 14.28 也无增益——旋转训练与丢弃竖排均非正贡献，
# 默认保留 baseline（v4 权重留档，见 STATUS「1.2 第 4 轮」节）。
RECOG_DEFAULT = PROJ / "pcbmodel" / "runs" / "ocr" / "recog_v1_anno1694_seed6_500ep" / "best.pt"
TILE = 1280
STRIDE = 768


@dataclass
class OcrText:
    bbox: list[float]          # y-up [x1, y1, x2, y2]
    cls: str                   # 7 类之一
    content: str
    conf: float
    raw: dict = field(default_factory=dict)   # 调试用：检测框/得分

    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


def _preprocess_crop(rgb: np.ndarray) -> np.ndarray:
    """灰度 → resize 高 32，与训练侧 `load_img(rotate=False)` 一致。

    ⚠ 2026-09-12 第 4 轮：去掉竖排旋转分支——训练已回滚旋转并丢弃竖排样本
    （端到端实测旋转负贡献：11.61 vs baseline 14.76），推理输入分布必须对齐。"""
    from PIL import Image
    from pcbmodel.ocr.train_recog import H
    im = Image.fromarray(rgb).convert("L")
    w, h = im.size
    nw = max(1, round(w * H / h))
    im = im.resize((nw, H), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def prop_pad(box, others=None, *, a: float = 0.15, lo: int = 1, hi: int = 4,
             frac: float = 0.4) -> int:
    """**按字高比例留白**（机制版，不是拍常数）→ 单边 pad 像素数。

    背景（2026-09-22）：val19 实测最优 = `2/13.1 ≈ 15%` 字高，而 golden 框高中位 22.1px
    ⇒ **固定 2px 跨尺度不转移**（在 golden 只占 9%，近乎无效）。故：

      `pad = clip(round(a·h), lo, hi)`，再用**到最近相邻文本框的间距**做上界（`frac` 倍）——
      后者是从机制上消掉"短位号吞邻居"（0032 型回退），不是再拍一个常数。

    `box`/`others` 都是 y-down 的 `(x1, y1, x2, y2)`（与 `_det_texts` 同帧）；**重叠的框跳过**
    （1.1e 对同一行字常出重叠双框，那不是"邻居"）。
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    h = max(1.0, y2 - y1)
    p = max(lo, min(hi, int(round(a * h))))
    gaps = []
    for o in (others or []):
        ox1, oy1, ox2, oy2 = (float(v) for v in o)
        dx = max(ox1 - x2, x1 - ox2, 0.0)
        dy = max(oy1 - y2, y1 - oy2, 0.0)
        if dx <= 0.0 and dy <= 0.0:
            continue                       # 与本框重叠 ⇒ 同一目标的重复框
        gaps.append(max(dx, dy))           # 轴上间距（文本框多按行/列排布）
    if gaps:
        p = min(p, max(0, int(frac * min(gaps))))
    return max(0, p)


def selftest() -> int:
    """比例留白自检（纯几何，不依赖 torch/数据集）。"""
    box = (0.0, 0.0, 30.0, 20.0)                       # h=20 ⇒ 0.15*20 = 3
    assert prop_pad(box, [], a=0.15) == 3, prop_pad(box, [], a=0.15)
    assert prop_pad(box, [], a=0.15, hi=2) == 2        # hi 可调
    assert prop_pad((0.0, 0.0, 10.0, 8.0), [], a=0.15) == 1      # lo 保护短位号
    assert prop_pad(box, [], a=0.30) == 4              # a*h=6 被 hi=4 截断
    assert prop_pad(box, [(34.0, 0.0, 60.0, 20.0)]) == 1         # 邻居 4px ⇒ 0.4*4=1
    assert prop_pad(box, [(32.0, 0.0, 60.0, 20.0)]) == 0         # 邻居 2px ⇒ 0（不吞邻居）
    assert prop_pad(box, [(5.0, 5.0, 25.0, 15.0)]) == 3          # 重叠框不算邻居
    assert prop_pad(box, [(0.0, 22.0, 30.0, 40.0)]) == 0         # 正下方 2px ⇒ 0
    print("[predict_ocr] prop_pad selftest OK（15% 字高 / clip 1..4 / 邻居上界 0.4×）")
    return 0


class OcrEngine:
    """进程内单例（runner 按例调用，权重加载一次）。必须 pcbmodel env（torch）。"""

    def __init__(self, det_weights: str | None = None, recog_ckpt: str | None = None,
                 device: str = "0", det_conf: float = 0.25, det_iou: float = 0.5,
                 pad_mode: str = "off", pad_a: float = 0.15,
                 crop_pad: int | None = None):
        from ultralytics import YOLO
        from pcbmodel.ocr.train_recog import CRNN, greedy_decode  # noqa: F401
        self.det_weights = str(det_weights or TEXT_DET_DEFAULT)
        self.recog_ckpt = str(recog_ckpt or RECOG_DEFAULT)
        if not Path(self.det_weights).is_file():
            raise FileNotFoundError(f"1.1e 检测权重不存在：{self.det_weights}")
        if not Path(self.recog_ckpt).is_file():
            raise FileNotFoundError(f"1.2 识别权重不存在：{self.recog_ckpt}")
        self.device = str(device)
        if self.device.isdigit():            # CLI 惯例传 "1" → torch 需要 "cuda:1"
            self.device = f"cuda:{self.device}"
        self.det_conf = det_conf
        self.det_iou = det_iou
        # "off"（默认 = 历史行为逐位不变）| "prop"（按字高比例留白，见 `prop_pad`）。
        # ⚠ 用 env 覆盖是为了**不动 runner 的 lru_cache 管道**就能跑 A/B（要进交付路径再加 CLI）：
        #   `PCB_OCR_PAD_MODE=prop PCB_OCR_PAD_A=0.15 python -m pcbmodel.integrate.ladder ...`
        import os as _os
        self.pad_mode = pad_mode if pad_mode != "off" else _os.environ.get("PCB_OCR_PAD_MODE", "off")
        self.pad_a = float(_os.environ.get("PCB_OCR_PAD_A", pad_a))
        # crop_pad: 在线推理从检测框裁字时外加的留白 px。
        # 训练 crop 用 pad=4（dataset_recog.crop_text_orig），旧推理 pad=0 造成 train/serve skew；
        # b 桶归因实测：位号命中 72.7%(pad0)->89.2%(pad2)，净救 47/286 回退 1。默认 0=现状，env PCB_CROP_PAD 覆盖。
        self.crop_pad = int(os.environ.get("PCB_CROP_PAD", "2")) if crop_pad is None else int(crop_pad)

        self.det = YOLO(self.det_weights)
        ck = torch.load(self.recog_ckpt, map_location="cpu")
        self.chars: list[str] = ck["vocab"]
        blank = len(self.chars)
        self.model = CRNN(blank + 1, hidden=ck["args"].get("hidden", 256))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

        vp = Path(TEXT_VOCAB_DEFAULT)
        if vp.is_file():
            self.idx2name = {int(v): k for k, v in json.loads(
                vp.read_text(encoding="utf-8"))["vocab"].items()}
        else:
            self.idx2name = {}
        self._greedy = greedy_decode

    def _det_texts(self, img: np.ndarray) -> list[dict]:
        """切块文本检测（与 `detect/eval_box.predict_case` 一致 + `max_det=2000`）。

        不能直接调 `predict_case`：它没传 `max_det`，ultralytics 默认 300 ——
        密集图（单块最多 488 个文本目标）会被截断，designator 类被高置信
        text_other 挤出（实测 0106 整例 designator=0）。max_det 只增模型输出
        上限，不改变模型容量。"""
        from pcbmodel.detect.tiles import (TilePlan, merge_across_tiles,
                                           plan_tiles, to_full_box)
        plan = TilePlan(TILE, STRIDE)
        H, W = img.shape[:2]
        tiles = plan_tiles(W, H, plan)
        dets: list[dict] = []
        for rect in tiles:
            x0, y0, x1, y1 = rect
            crop = img[y0:y1, x0:x1]
            res = self.det.predict(crop, imgsz=plan.tile, conf=self.det_conf,
                                   iou=self.det_iou, device=self.device,
                                   verbose=False, max_det=2000)
            for r in res:
                if r.boxes is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                sc = r.boxes.conf.cpu().numpy()
                for b, c, s in zip(xyxy, cls, sc):
                    norm = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
                    nb = [(norm[0] + norm[2]) / 2 / tw, (norm[1] + norm[3]) / 2 / th,
                          (norm[2] - norm[0]) / tw, (norm[3] - norm[1]) / th]
                    dets.append({"box": to_full_box(nb, rect), "score": float(s),
                                 "cls": int(c)})
        return merge_across_tiles(dets)

    def _recog_conf(self, logits: torch.Tensor) -> float:
        """CTC 贪心解码路径的**每字符平均 log-prob**（识别置信度）。

        ⚠ 2026-09-18（B2，Claude 清单第 1 条）：`OcrText.conf` 原是 1.1e 检测框
        分数，不是识别置信度；`match_13` 冲突只按几何距离 tiebreak。本方法就地
        复算贪心路径（与 `greedy_decode` 同折叠规则）的平均 log-prob，**不改任何
        行为**，只把数据留给后续 match 消歧。"""
        T, B, C = logits.shape
        blank = C - 1
        logp = logits.log_softmax(-1)
        conf: list[float] = []
        for b in range(B):
            probs, ids = logp[:, b].max(-1)
            prev = blank
            for t in range(T):
                c = int(ids[t])
                if c != blank and c != prev:
                    conf.append(float(probs[t]))
                prev = c
        return float(sum(conf) / len(conf)) if conf else 0.0

    def predict_case(self, case_id: str, source: str = "official-train") -> list[OcrText]:
        from pcbmodel.data.loader import load_sample

        s = load_sample(case_id, source=source, with_mask=False)
        dets = self._det_texts(s.image_orig)
        ih, iw = s.image_orig.shape[:2]
        out: list[OcrText] = []
        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
            # ⚠⚠ 单边留白**只算一次**：`prop` 是**替换** `crop_pad`，不是叠加。
            #   2026-09-22 自检抓到叠加版（先 +2、再在**已扩框**上算 prop）⇒
            #   总留白 = 2 + p；且 `h` 被放大 4px、邻居间距被压小 4px。
            #   在 golden(22px 框) 上变成 ~6px 过补 ⇒ −1.26 分。**h 与邻居距离都必须用原始 det 框算。**
            if self.pad_mode == "prop":
                p = prop_pad((x1, y1, x2, y2), [o["box"] for o in dets if o is not d],
                             a=self.pad_a)
            else:
                p = self.crop_pad
            x1, y1 = max(x1 - p, 0), max(y1 - p, 0)
            x2, y2 = min(x2 + p, iw), min(y2 + p, ih)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = s.image_orig[y1:y2, x1:x2]
            arr = _preprocess_crop(crop)
            if arr.shape[1] < 4:                 # CRNN 时间步 T=nw/4，nw<4 无输出
                continue
            x = torch.from_numpy(arr[None, None]).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
            ids = self._greedy(logits)[0]
            content = "".join(self.chars[c] for c in ids if c < len(self.chars))
            recog_conf = self._recog_conf(logits)   # B2：识别置信度（零行为改变）
            cls = self.idx2name.get(int(d.get("cls", -1)), "text_other")
            out.append(OcrText(
                bbox=[float(x1), float(s.H - y2), float(x2), float(s.H - y1)],
                cls=cls, content=content, conf=float(d.get("score", 0.0)),
                raw={"det_box": list(d["box"]), "det_cls": int(d.get("cls", -1)),
                     "recog_conf": recog_conf}))
        return out
