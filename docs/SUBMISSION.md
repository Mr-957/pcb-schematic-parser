# 比赛提交与本地容器运行

1. 获取官方 `edarace-agent-base:ubuntu24`，用官方提供的镜像导入方式加载。
2. 获取 `offline_wheels.zip`（与本版本的 `offline_wheels_manifest.json` 一致），在仓库根执行：

```bash
python scripts/prepare_offline.py /path/to/offline_wheels.zip
python scripts/check_assets.py
docker build -t pcb-schematic-parser:20261009 .
docker run --rm   -v /absolute/path/to/pngs:/input:ro   -v /absolute/path/to/results:/output:rw   pcb-schematic-parser:20261009 -input /input -output /output
```

3. 检查 `decision_log.json` 的 `n_error=0`、案例数量与输入图片数量一致，以及每张图片对应的 `_Pred.json`。容器输出还包含启动诊断和完整 stdout/stderr。
4. 打包上传：

```bash
python scripts/package_submission.py --output dist/submission.zip
```

离线依赖只适用于官方 Ubuntu24/Python3.12/Linux x86_64 环境；不能用于 Windows 或 Python3.11。Dockerfile 沿用已跑通平台基线的离线安装策略，保留基础镜像的 PyTorch/CUDA，使用 `--break-system-packages` 和 `--no-index`，防止构建过程中因联网下载超时失败。

当前服务器没有 Docker socket 使用权限，所以本次没有完成 Docker build/run 验证。普通 Python 的真实 CPU 推理验证不能替代官方容器验证。本仓库整理后的新 ZIP 也需要重新验证，不能直接继承上一份 ZIP 的验收结论。
