# PCB Schematic Parser

将 PCB 原理图 PNG 解析成器件、引脚和网络 JSON。纯 CV 推理：YOLO 元件/文本检测 → CRNN OCR → U-Net 导线分割 → 规则拓扑与 box 引脚解析。不调用大模型 API。

本仓库提供可直接使用的推理代码、四份模型权重、类别词表和 IC 运行库；不包含训练/验证图片、GT、实验输出、内部交流或交接文档。重点是推理和比赛提交，不提供完整训练数据及训练流程。

## 快速开始

建议 Linux x86_64、Python 3.12。先将仓库完整克隆或下载解压；四份权重随仓库提供，不需要 Git LFS。运行只需输入图片，不需要 GT。

```bash
git clone https://github.com/Mr-957/pcb-schematic-parser.git
cd pcb-schematic-parser
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/check_assets.py
python agent.py -input /absolute/path/to/pngs -output /absolute/path/to/results --device cpu
```

CPU 安装命令来源：[PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。需要 GPU 时，将安装命令的 `cpu` 索引替换为官方对应的 `cu121` 索引，然后用 `--device 0`。入口会实际检查 GPU 是否可用，不支持的架构或初始化失败会回落 CPU。

输入目录平铺 PNG 等支持的图片；推荐 PNG，每张图片生成一份预测。输入路径、输出路径可以含空格，请加引号。Windows 使用 Python 入口，不运行 Bash 脚本；Windows 和全新环境安装尚未实测。

## 输出

```text
results/
├── final_results/<图片文件名>_Pred.json
├── decision_log.json
├── desig_reclass.json
└── name_fill.json
```

预测顶层为 `components`、`pins`、`nets`，坐标以图片左下角为原点、y 向上。`decision_log.json` 记录设备、耗时、成功/失败案例以及纯 CV 的零 API token 用量；请检查 `n_error`，不能仅凭进程退出码认定所有图片成功。

## 当前配置

- `predictor=ocr-real`、`boxes=detect`、`nets=wire`、`box_pins=wire`；pin-anchor 关闭。
- 位号纠正 `PCBM_DESIG_RECLASS=1` 和 IC 型号补全 `PCBM_NAME_FILL=1` 在 `agent.py` 中默认启用。Python 入口允许在启动前将环境变量设为 `0`；比赛 `run.sh` 强制启用两项。
- 官方 IC 库默认使用仓库内文件；可用 `PCB_IC_LIB` 指定其他路径。
- 导线主动延伸等未晋升实验未启用。
- 算法及权重沿用已审核联合候选。仅入口默认配置、IC 库路径和仓库/打包结构做了可移植适配。

## 目录

```text
agent.py                    Python 推理入口
run.sh                      比赛 Bash 入口与诊断日志
Dockerfile                  官方比赛镜像构建文件
requirements.txt            普通 Python 安装依赖
pcbmodel/                   推理及其依赖模块、权重/词表
pcbeval/                    推理所依赖的坐标、引用与配置工具
10gt/10gt/                  官方 IC 运行库（不包含评测图片/GT）
scripts/                    资产校验、离线依赖准备、提交打包
assets_manifest.json        运行资产 SHA256
offline_wheels_manifest.json 离线依赖 SHA256
docs/                       使用与验证说明
```

部分模块名称含 `train`、`eval`、`probe`，但它们含有当前模型结构或其他模块的导入依赖，因此保留；本仓库不承诺可复现完整训练/历史实验。`pcbeval` 是项目内部工具，不能将其本地分数解释为官方平台分数。

## 比赛提交

普通 Python 推理不需要平台镜像或离线 wheelhouse。比赛镜像基于官方 `edarace-agent-base:ubuntu24`；镜像需从官方渠道获取，本仓库没有提供该基础镜像。

离线依赖约 248 MB，独立作为 `offline_wheels.zip` 发布附件，不放进 Git 源码历史。也可直接使用已审核的 `submission_20261009_desig_name.zip` 提取相同依赖：

```bash
python scripts/prepare_offline.py /path/to/offline_wheels.zip
# 或：python scripts/prepare_offline.py /path/to/submission_20261009_desig_name.zip
python scripts/package_submission.py --output dist/submission.zip
```

ZIP 根目录直接包含 `Dockerfile`、`agent.py`、`run.sh`，不再嵌套一层仓库文件夹。详细构建/容器运行命令见 [比赛提交说明](docs/SUBMISSION.md)。GitHub 的源码 Download ZIP 带仓库父目录且没有 wheelhouse，不能直接上传比赛平台。

## 验证与结果范围

参见 [验证记录](docs/VALIDATION.md)。已知 48.04 是此前离线基线在官方平台上的得分；本仓库加入位号纠正和 IC 型号补全，不将基线分数当作本版本官方成绩。新仓库还需要实际平台提交验证。

## 代码与资产来源

代码由本项目整理；模型来自项目训练并与已审核提交包保持一致。IC 库来自赛题官方公开资料，`ic_pin_db.json` 为既有运行时投票表；推理不读取输入案例 GT，不会现场重建该表。具体第三方依赖见 [第三方说明](THIRD_PARTY.md)。本次尚未给整个项目指定统一开源许可证；公开访问不代表已授予任意再许可。
