# 第三方与资产来源

本项目使用 PyTorch / torchvision、Ultralytics、NumPy、SciPy、Pillow、scikit-image、OpenCV 等第三方软件。依赖由各自官方发行包安装，它们适用各自许可证；本项目没有将第三方依赖重新声明为本项目许可证。

- PyTorch: https://github.com/pytorch/pytorch
- Ultralytics: https://github.com/ultralytics/ultralytics
- scikit-image: https://github.com/scikit-image/scikit-image
- OpenCV: https://github.com/opencv/opencv
- 官方 IC 库：赛题六公开的 `BoxICLibrary_200cases.json`，随项目保留原文件，不改写内容。
- 模型权重与类别词表：本项目现有推理权重，与已审核联合候选包一致；校验值见 `assets_manifest.json`。
- `ic_pin_db.json`：现有管线依赖的既有静态投票表，原样保留。本次整理不重新生成、不引入评测 GT。

本仓库尚未声明统一开源许可证；使用者应分别遵守依赖、官方资料及赛事对应条款。
