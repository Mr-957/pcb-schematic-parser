# EDARace 交付镜像 —— **薄层**：基座是官方的，我们只加依赖 + 代码 + 权重 + 入口。
#
# 构建（在**仓库根** `/home/linqs/code/CV for PCB`，不是本目录）：
#   docker build -f pcbmodel/Dockerfile -t <队伍镜像名> .
# 官方契约见 edarace-agent-base_ubuntu24/extracted/submission/README.md（本文件逐条对应）。
#
# ⚠ 官方基础镜像**不含** ultralytics / scikit-image（README 的组件表里没有）⇒ 必须自装。
#    官方镜像自带 opencv-python-**headless**；ultralytics 会拉非 headless 的同名 `cv2`，
#    两者共存，靠安装顺序定胜负 —— 故这里**显式重装 headless 版压回**（无 GUI，容器里本就不需要）。
FROM edarace-agent-base:ubuntu24

# ⚠ 评测容器**可能没有外网**。ultralytics 缺包时会自己 pip install（实测喂一张坏图即触发
#    `pi-heif` 下载、卡 7.6s）⇒ 从镜像层就关掉，别指望运行时兜。
ENV YOLO_AUTOINSTALL=false \
    PYTHONPATH=/submission \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8 \
    PCBM_DESIG_RECLASS=1 \
    PCBM_NAME_FILL=1

# 【联合候选 2026-10-09】两个**已审核**候选开关在此显式打开（用户指令"显式设置"）：
#   PCBM_DESIG_RECLASS=1  位号校正（把 text_other 中明确位号收回 designator）
#   PCBM_NAME_FILL=1      给空 Name 的 box 补"官方库确认 + OCR 原文"的 IC 型号
# 两者**互相独立**，都写在镜像 ENV 里 ⇒ `docker inspect` 可见、且任何绕过 run.sh 的调用式
# （例如直接 `python3 /submission/agent.py`）也带着它们。run.sh 里另有一份同名 export，
# 保证**解压出来直接跑**同样生效。默认关闭的生产行为**不在本包内** —— 这是独立候选包，
# 不覆盖基线包。box 引脚链与导线后处理**仍是基线**，未启用任何主动延伸/交叉点/补脚/pin-anchor。

# 官方 IC 库（box 引脚链的唯一名判据）。`pin_from_name.LIB_PATH` 默认是**服务器绝对路径**，
# 容器里没有那个路径 ⇒ 用 `PCB_IC_LIB` 指到镜像内这一份，**算法一行不改**（Codex 审核 R1）。
# 库原件来自 `10gt/10gt/BoxICLibrary_200cases.json`（md5 a163e4b1343289f6ead6abbda912ad5a，
# 1826150 bytes）；`10gt/` 的**图与 GT 都不进上下文**，见 .dockerignore。
ENV PCB_IC_LIB=/submission/10gt/10gt/BoxICLibrary_200cases.json
COPY 10gt/10gt/BoxICLibrary_200cases.json /submission/10gt/10gt/BoxICLibrary_200cases.json

# 与官方镜像同规格的推理依赖。**锁版本**：ultralytics 8.4.146 = 两个 YOLO checkpoint
# 内标的训练版本（换版本 = 换反序列化路径，没验证过）。
# 问题4修复：拆成两步——先装 ultralytics/scikit-image，再单独 force-reinstall headless cv2
# 原因：同一条 RUN 里 pip 判 already satisfied 不会重装 headless，而 ultralytics 依赖非 headless
# 的 opencv-python 会覆盖安装，其 cv2.abi3.so 链 libGL.so.1，基座镜像里没有 GL 库 ⇒ import cv2 直接炸。
#   ⇒ 装完之后 `opencv-python` 的 dist-info 仍在（headless 是另一个发行名，卸载不到它），
#     这是**有意保留**的：ultralytics 元数据里写的依赖就是 `opencv-python`，删掉 pip check 会报缺包。
# ⚠ **必须 `python3 -m pip ... --break-system-packages`**（2026-10-08 第二次平台提交实测）：
#   Ubuntu 24.04 的系统 Python 按 PEP 668 标记为 externally-managed，裸 `pip install` 会被
#   直接拒绝 —— 平台日志 Step 5/22 报 `error: externally-managed-environment`，**构建失败**，
#   镜像没起来、入口与模型都没运行（`summary.json not found` 只是它的连带结果）。
#   基座镜像未建 venv，官方样例的自定义结构示例同样用 `pip3 install --break-system-packages`。
#
# ⚠ **离线安装：`--no-index --find-links`**（2026-10-09 第四次实测，提交 ID 4516）：
#   第三次显式加 `--index-url https://pypi.org/simple` 后，pip **把解析走完了**——日志里
#   matplotlib 3.11.2 / scipy 1.18.1 / scikit-image 0.26.0 的 cp312 wheel 都拿到了，
#   基座的 torch 2.5.1+cu121 / torchvision 0.20.1+cu121 / numpy 2.5.3 全部 already satisfied。
#   然后**下载第一个 wheel 就断了**：`ultralytics-8.4.146-py3-none-any.whl` 读到 0.2/1.4 MB，
#   14.0 kB/s，`ReadTimeoutError` @ files.pythonhosted.org ⇒ 构建 exit 2。
#   ⇒ 结论：平台**能**连 PyPI，是链路速度/稳定性撑不住；加大 timeout 只能缓解、不能保证。
#     故改为**把依赖整体打进 ZIP**：`submission_wheels/` 的 26 个 wheel 全部是平台那次
#     真实解析选中的文件（清单见 MANIFEST.json 的 url/sha256），构建期**零网络下载**。
#   不加 `--trusted-host`、不关 TLS、不给 ultralytics 加 `--no-deps`、不替换官方 torch/CUDA。
# Diagnostics run before dependency installation. No full environment dump.
COPY pcbmodel/integrate/submission_diag.py /submission/submission_diag.py
COPY pcbmodel/integrate/submission_constraints.txt /submission/submission_constraints.txt
COPY submission_wheels/ /submission/submission_wheels/
RUN python3 -u /submission/submission_diag.py --phase build --root /submission

# Preserve complete resolver output and the original pip failure status.
# Constraints pin the exact wheel set, so the resolver has no version freedom left.
RUN bash -o pipefail -c 'python3 -m pip install -v --no-cache-dir --break-system-packages --no-index --find-links /submission/submission_wheels --constraint /submission/submission_constraints.txt "ultralytics==8.4.146" "scikit-image==0.26.0" 2>&1 | tee /submission/build_pip_install.log'
RUN bash -o pipefail -c 'python3 -m pip install -v --no-cache-dir --break-system-packages --no-index --find-links /submission/submission_wheels --constraint /submission/submission_constraints.txt --force-reinstall --no-deps "opencv-python-headless==4.10.0.84" 2>&1 | tee /submission/build_cv2_install.log' \
    && python3 -c "import torch, ultralytics, skimage, cv2; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| skimage', skimage.__version__, '| cv2', cv2.__version__)"

# Advisory only: `pip check` must not be the thing that fails the build on the last upload.
# (It is expected to pass — opencv-python stays registered alongside the headless cv2.)
RUN bash -o pipefail -c 'python3 -m pip check 2>&1 | tee /submission/build_pip_check.log; echo "[build] pip check above is advisory; install/import checks above are the gate"'

WORKDIR /submission

# 代码（`.dockerignore` 已把 runs/ 的大件、数据集、镜像 tar 全排掉）
COPY pcbmodel/ /submission/pcbmodel/
COPY pcbeval/ /submission/pcbeval/

# 权重：只有这 5 个文件是推理必需的（≈147MB + 两个小词表）
COPY pcbmodel/runs/detect/train/yolo11m_v2/weights/best.pt \
     /submission/pcbmodel/runs/detect/train/yolo11m_v2/weights/best.pt
COPY pcbmodel/runs/detect/train/text_v1/weights/best.pt \
     /submission/pcbmodel/runs/detect/train/text_v1/weights/best.pt
COPY pcbmodel/runs/seg/unet_v1/best.pt \
     /submission/pcbmodel/runs/seg/unet_v1/best.pt
COPY pcbmodel/runs/ocr/recog_v1_anno1694_seed6_500ep/best.pt \
     /submission/pcbmodel/runs/ocr/recog_v1_anno1694_seed6_500ep/best.pt
COPY pcbmodel/runs/detect/ds_v2/vocab.json \
     /submission/pcbmodel/runs/detect/ds_v2/vocab.json
COPY pcbmodel/runs/ocr/ds_text_v1/vocab.json \
     /submission/pcbmodel/runs/ocr/ds_text_v1/vocab.json

# 入口：**两个名字**，都只转发 `deliver`（契约同为 `-input <dir> -output <dir>`）。
#   ① `/submission/agent_system` —— 覆盖官方 placeholder，也是基座自带 ENTRYPOINT 的名字；
#   ② `/submission/run.sh`      —— 2026-10-08 新提交样例要求的名字（ENTRYPOINT 指向它）；
#   ③ `/submission/agent.py`  —— 2026-10-08 官方「方式1：纯 CV」样例的**根目录入口名**，
#      与 run.sh **同目录**，run.sh 调它，它再转 `deliver.main()`（薄转发，无算法）。
# run.sh 容忍多余位置参数，所以新样例调用式与基座老调用式**都能跑**，见 run.sh 头部。
COPY pcbmodel/integrate/agent_system /submission/agent_system
COPY run.sh /submission/run.sh
COPY agent.py /submission/agent.py
RUN chmod +x /submission/agent_system /submission/run.sh /submission/agent.py

# 构建期自检：权重在位 + 三个入口都在 + run.sh 语法 + **agent.py 真能走到 deliver**（失败即构建失败）
RUN test -x /submission/agent_system && test -x /submission/run.sh && test -x /submission/agent.py \
    && bash -n /submission/run.sh \
    && python3 /submission/agent.py --help > /dev/null \
    && python3 -c "import pathlib,sys; \
need=['detect/train/yolo11m_v2/weights/best.pt','detect/train/text_v1/weights/best.pt', \
      'seg/unet_v1/best.pt','ocr/recog_v1_anno1694_seed6_500ep/best.pt', \
      'detect/ds_v2/vocab.json','ocr/ds_text_v1/vocab.json']; \
miss=[p for p in need if not pathlib.Path('/submission/pcbmodel/runs',p).is_file()]; \
sys.exit('缺权重/词表: %s' % miss if miss else 0)" \
    && python3 -c "import pcbmodel.integrate.deliver as d; \
print('入口 OK ｜ 交付配置', d.BEST)" \
    && python3 -c "from pcbmodel.integrate import pin_from_name as p; \
lib = p.load_official(); \
print('IC 库 OK ｜', p.LIB_PATH, '｜', len(lib), '个型号')" \
    && python3 -c "import pcbmodel.ocr.match_13 as m, pcbmodel.integrate.name_fill as nf; \
assert m.desig_reclass_on() and nf.name_fill_on(), '候选开关未生效'; \
print('候选开关 OK ｜', m.DESIG_RECLASS_ENV, '=on ｜', nf.NAME_FILL_ENV, '=on')"

# ENTRYPOINT：按 2026-10-08 新样例。基座原本是 `["/submission/agent_system"]`
# （`skopeo inspect --config oci:edarace-agent-base_ubuntu24:ubuntu24` 已核），指到 run.sh 后
# 老调用式（命令里带 `/submission/agent_system`）仍由 run.sh 的容错分支接住。
# 官方 config 里的 WorkingDir 已是 /submission，且 ENV PYTHONPATH=/submission（见上）。
ENTRYPOINT ["/submission/run.sh"]
