#!/usr/bin/env bash
# EDARace 提交入口 —— 2026-10-08 官方新样例要求：ZIP 根有 `run.sh`（可执行）+ `ENTRYPOINT ["/submission/run.sh"]`。
#
# 与 `/submission/agent_system` 的关系：两者都只转发 `python3 -m pcbmodel.integrate.deliver`。
# 官方基础镜像自带的 ENTRYPOINT 是 `["/submission/agent_system"]`（已核 OCI config），新样例改成 run.sh
# ⇒ 两个名字都保留，且本脚本**容忍多余位置参数**（未知参数丢弃）：
#     docker run <img> -input /input -output /output                          → 正常
#     docker run <img> /submission/agent_system -input /input -output /output → 也正常
# 第二形态并非臆测：ENTRYPOINT 会与命令行参数拼接，平台若沿用基座的老调用式就会长成这样，
# 不容忍就是整批 0 分且不报错。
set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -input|--input)   INPUT_DIR="${2:-}";  shift 2 ;;
    -output|--output) OUTPUT_DIR="${2:-}"; shift 2 ;;
    *) echo "[run.sh] 忽略未识别参数: $1" >&2; shift ;;
  esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: /submission/run.sh -input <input_dir> -output <output_dir>" >&2
  exit 2
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[run.sh] input 目录不存在: $INPUT_DIR" >&2
  exit 3
fi

# 【联合候选 2026-10-09】两个**已审核**候选开关在此显式打开 —— 与镜像 ENV 同值，重复一份是为了
# 让「解压这个 ZIP 直接跑」也带上它们（验收就是从解压内容跑的）。默认关闭的生产行为不在本包内。
export PCBM_DESIG_RECLASS=1     # 位号校正候选（上一轮已验收）
export PCBM_NAME_FILL=1         # IC 型号 name 补填候选（本轮已验收）
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 启动自检：box 引脚链的官方 IC 库必须在位（Codex 审核 R1）。**只吼不中止** ——
# 「库缺失」的失败形态是相关例子**静默少脚**而非报错，所以必须显式留痕；但整批 abort 的代价
# （0 分）大于让其余例子跑完，故这里不做硬失败。硬失败放在构建期与验收脚本里。
if ! ICMSG=$(PYTHONPATH="$SELF_DIR${PYTHONPATH:+:$PYTHONPATH}" \
             python3 -c "from pcbmodel.integrate.pin_from_name import LIB_PATH; print(LIB_PATH)" 2>&1); then
  echo "[run.sh] ⚠ IC 库自检执行不了: $ICMSG" >&2
elif [ ! -f "$ICMSG" ]; then
  echo "[run.sh] ⚠ IC 库不在位: $ICMSG（PCB_IC_LIB=${PCB_IC_LIB:-未设}）—— box 引脚链会失败" >&2
fi

mkdir -p "$OUTPUT_DIR/final_results"
# 转调**同目录的** agent.py（官方方式1 要求的根目录入口），由它转 `deliver.main()`。
# 用 $SELF_DIR 而不是 `/submission/agent.py`：本地仓库根、容器 /submission、以及解到任意临时
# 目录都能跑（这也是验收脚本「解出包直接跑」的前提）。
# Save startup diagnostics and full stdout/stderr under the mounted output.
export PYTHONUNBUFFERED=1
python3 -u "$SELF_DIR/pcbmodel/integrate/submission_diag.py" --phase startup \
  --root "$SELF_DIR" --input "$INPUT_DIR" --output "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/startup_diagnostics.log" || true
set +e
python3 -u "$SELF_DIR/agent.py" -input "$INPUT_DIR" -output "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/agent_stdout_stderr.log"
RUN_STATUS=("${PIPESTATUS[@]}")
set -e
echo "[run.sh] agent exit=${RUN_STATUS[0]}, log writer exit=${RUN_STATUS[1]}" \
  | tee -a "$OUTPUT_DIR/agent_stdout_stderr.log"
if [[ "${RUN_STATUS[0]}" -ne 0 ]]; then exit "${RUN_STATUS[0]}"; fi
exit "${RUN_STATUS[1]}"
