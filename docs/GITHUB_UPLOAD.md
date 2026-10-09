# 发布到 GitHub

目标：`Mr-957/pcb-schematic-parser`，Public。

本地准备目录：`D:\Match\CV for PCB\github`。其中 `pcb-schematic-parser` 是源码仓库，`offline_wheels.zip` 是独立离线依赖附件，`publish_to_github.ps1` 是本地发布辅助脚本。发布脚本不包含密码或令牌。

## 一次性发布

Windows PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Match\CV for PCB\github\publish_to_github.ps1"
```

1. 脚本打开 GitHub 官方登录授权流程；使用 Mr-957 登录并完成授权。无需向 Codex 发送密码、令牌或验证码。
2. 脚本核对登录账号，若目标仓库不存在则创建 Public 仓库；若存在则复用公开仓库。
3. 在本地提交并推送 main；设置仓库级的 GitHub noreply 提交邮箱，保留你全局 Git 配置。
4. 创建 v2026.10.09 预发布，并上传约248 MB 的离线依赖附件。

脚本没有强制推送、删除仓库或覆盖已有发布附件的操作。如目标仓库已有不兼容历史，push 会失败，请交回 Codex 处理。如网络超时，恢复 GitHub 网络访问后重试；已完成的提交和发布会被复用。

## 手动操作备选

登录 https://github.com/new ：Owner 选 Mr-957，仓库名填 pcb-schematic-parser，选择 Public；不要预先添加 README、gitignore 或 License，点击 Create repository。再使用上述发布脚本完成推送及附件上传。

普通推理的四份权重已经在源码内，不需要 Git LFS。离线附件仅用于官方 Ubuntu24/Python3.12/Linux x86_64 比赛镜像。不要将 GitHub Download ZIP 直接当比赛提交包上传。

发布完成后，将 https://github.com/Mr-957/pcb-schematic-parser 发给队友即可。公开仓库无需邀请协作者即可查看、下载和克隆；写入代码的权限仍需要单独授权。
