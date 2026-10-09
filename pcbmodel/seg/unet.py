# -*- coding: utf-8 -*-
"""P1 阶段 2.1：U-Net 导线分割 —— 细线必须有**全分辨率 skip**。

**为什么不用 torchvision 的 deeplabv3/fcn（实测依据，不是偏好）**：
官方 `nets.edges` 光栅化后**平均线宽 2.07px**（8 例实测 2.01–2.28 = `掩码像素/骨架像素`）。
`deeplabv3_resnet50` / `fcn_resnet50` 输出 stride 8 —— 一条 2px 的线在 stride-8 特征图上
**不足一个格**，结构上不可分辨。故必须自写 U-Net：编码器逐级下采样取上下文，
解码器**逐级 skip 回 stride 1**，输出与原图同分辨率。（本机无 `segmentation_models_pytorch`。）

**帧**：全部在 `sample.image_up` / `sample.wire_mask_up` 的 canonical **y-up** 帧里做，
`arr[y, x]` ↔ 坐标 `(x, y)`，**不翻帧**。与 OCR 线相反 —— 那里必须 y-down 才让文字正立；
分割对方向不敏感，翻帧只会多一处出错点。（对比 `detect/tiles.py` 的 y-down 约定。）

**⚠ 验收口径不是 Dice/IoU**：下游是 `skeletonize → RDP → 2 点段`（`wire/ceiling.py`）。
2px 的线偏 1px，Dice 只有 ~0.33，但骨架**拓扑完全正确** ⇒ 下游照样满分。
所以 Dice 只当日志诊断，**选模型/验收一律看 `netF1@5px`**（见 `train_seg.py --eval-every`）。

运行（**pcbmodel env**，本模块要 torch）：
  <env>/bin/python -m pcbmodel.seg.unet --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根

# 切块起点规则与 `detect/tiles.py` 逐字一致（**本项目同一条覆盖保证**：并集恰好覆盖
# `[0, total]`，末块贴边）。直接复用而不复制 —— 两条线同一个作者，改一处即可。
from pcbmodel.detect.tiles import _axis_starts


def img_to_tensor(img: np.ndarray):
    """[H,W,3] uint8 → [1,3,H,W] float32/255。**训练与推理共用同一个归一化**，别各写一份。"""
    import torch
    a = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(a).unsqueeze(0)


class _DoubleConv:
    """两层 3×3 + BN + ReLU。用函数式工厂避免多写一个 Module 类。"""

    @staticmethod
    def make(cin: int, cout: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))


def build_unet(base: int = 32, depth: int = 4, in_ch: int = 3):
    """默认 base=32 / depth=4 ⇒ **7.85M**（实测）。

    为什么不是 depth=5（512 输入下 32× 下采样）：瓶颈变成 16×16 且 512→1024 那一层
    独占一半参数（depth5/base32 = 31.4M）—— 而本任务要的是 **2px 线的细节**，
    容量该留在高分辨率层。depth=4 的 16× 下采样感受野已足够（512→32×32 瓶颈）。
    训练集只有 **171 例**，7.85M 比 31.4M 稳妥得多。

    `crop` 必须是 `2**depth` 的整数倍（512 % 16 == 0），否则下采样/上采样后 skip 对不齐。
    """
    import torch
    import torch.nn as nn

    chs = [base * 2 ** i for i in range(depth)]

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.ModuleList()
            c = in_ch
            for ch in chs:
                self.enc.append(_DoubleConv.make(c, ch)); c = ch
            self.pool = nn.MaxPool2d(2)          # depth 段之间 depth-1 次 + 进瓶颈 1 次
            self.bott = _DoubleConv.make(chs[-1], chs[-1] * 2)
            self.up, self.dec = nn.ModuleList(), nn.ModuleList()
            prev = chs[-1] * 2
            for ch in reversed(chs):
                self.up.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                self.dec.append(_DoubleConv.make(prev + ch, ch))
                prev = ch
            self.head = nn.Conv2d(prev, 1, 1)

        def forward(self, x):
            skips = []
            for k, e in enumerate(self.enc):
                if k:                            # 第一个编码段不下采样
                    x = self.pool(x)
                x = e(x); skips.append(x)
            x = self.bott(self.pool(x))
            for u, d, s in zip(self.up, self.dec, reversed(skips)):
                x = d(torch.cat([u(x), s], dim=1))
            return self.head(x)

    return UNet()


def load_ckpt(path, device: str = "cuda"):
    """读回 `train_seg.py` 存的权重（只存 state_dict + 结构超参，不 pickla 整个模型）。"""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = build_unet(base=blob.get("base", 32), depth=blob.get("depth", 4))
    net.load_state_dict(blob["state_dict"])
    return net.to(device).eval()


def predict_prob(model, image_up: np.ndarray, *, crop: int = 512, stride: int = 384,
                 device: str = "cuda", batch: int = 8, amp: bool = True) -> np.ndarray:
    """滑窗预测**全图概率图**（float32 [H,W]）。

    重叠区按**概率平均**（不是按 logits 相加，也不是硬拼）—— 压制接缝台阶。
    比 `crop` 小的图先 reflect pad 到 `crop`：否则末块尺寸不等于 crop，
    解码器逐级 ×2 会对不齐 skip 而 shape 崩。

    与 `predict_mask` 分开是刻意的：**阈值是一条要扫的轴**（ceiling 报告结论②：
    后处理该按连通性/段数扫，不该按 IoU 调），扫阈值时不该把推理重跑一遍。
    """
    import torch
    h, w = image_up.shape[:2]
    ph, pw = max(crop, h), max(crop, w)
    img = image_up
    if (ph, pw) != (h, w):
        img = np.pad(image_up, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")

    pos = [(y, x) for y in _axis_starts(ph, crop, stride) for x in _axis_starts(pw, crop, stride)]
    acc = np.zeros((ph, pw), np.float32)
    cnt = np.zeros((ph, pw), np.float32)
    model.eval()
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        # 切片是 view（不复制）；只有 np.stack 这一小批会复制
        b = np.stack([img[y:y + crop, x:x + crop] for y, x in chunk])
        t = torch.from_numpy(np.ascontiguousarray(b.transpose(0, 3, 1, 2))
                             ).float().div_(255.0).to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=amp and device.startswith("cuda")):
            logit = model(t)
        probs = torch.sigmoid(logit.float())[:, 0].cpu().numpy()
        for (y, x), p in zip(chunk, probs):
            acc[y:y + crop, x:x + crop] += p
            cnt[y:y + crop, x:x + crop] += 1.0
    return acc[:h, :w] / np.maximum(cnt[:h, :w], 1.0)


def predict_mask(model, image_up: np.ndarray, *, thr: float = 0.5,
                 **kw) -> np.ndarray:
    """`predict_prob` 再阈值 → y-up uint8 0/1 掩码。"""
    return (predict_prob(model, image_up, **kw) >= thr).astype(np.uint8)


def selftest() -> int:
    """形状/参数量/滑窗覆盖/小图 pad —— 不训不评，只保证这几条硬约束。"""
    import torch
    net = build_unet()
    n = sum(p.numel() for p in net.parameters())
    x = torch.zeros(2, 3, 512, 512)
    y = net(x)
    assert y.shape == (2, 1, 512, 512), y.shape
    print(f"[seg.unet] 前向 OK {tuple(y.shape)}  参数量 {n/1e6:.2f}M")

    # 滑窗：重叠区必须被平均（cnt>1），且并集完整覆盖 → 全图无 hole
    img = np.zeros((900, 1300, 3), np.uint8)
    net.eval()
    with torch.no_grad():
        net.head.bias.fill_(5.0)          # 常量正输出 → 掩码应全 1（证明覆盖无洞）
    m = predict_mask(net, img, crop=512, stride=384, device="cpu", amp=False, batch=4)
    assert m.shape == (900, 1300) and m.all(), f"覆盖有洞或尺寸错：{m.shape} {m.mean()}"
    print("[seg.unet] 滑窗覆盖 OK 900x1300 全 1（无洞）")

    # 比 crop 小的图：pad 后仍要出原尺寸
    m2 = predict_mask(net, np.zeros((300, 481, 3), np.uint8), crop=512, stride=384,
                      device="cpu", amp=False)
    assert m2.shape == (300, 481) and m2.all(), m2.shape
    print("[seg.unet] 小图 pad OK 300x481 → 原尺寸")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seg.unet", description="U-Net 导线分割模型")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
