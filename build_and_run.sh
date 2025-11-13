#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ================================
# 构建阶段（原 build.sh 内容）
# ================================
echo "=== Building project ==="
swgcc -mslave -O3 -msimd -funroll-loops slave.c -c -o slave.o #-funroll-loops
swgcc -faddress_align=128 -mhost -msimd -O3 master.c -c -o master.o
swg++ -faddress_align=128 -mhost -msimd -O3 main.cpp -c -o main.o
swg++ -mhybrid -static main.o master.o slave.o -o exe
echo "=== Build finished ==="


# ================================
# 提交阶段（原 run2.sh 内容）
# ================================
QUEUE="q_sw_expr"
RESULT_DIR="./results"
DATE_DIR="$RESULT_DIR/$(date +"%Y%m%d")"
mkdir -p "$DATE_DIR"

EXEC_FILES=(
    "exe"
)

echo "=== Submitting jobs ==="
for EXEC_FILE in "${EXEC_FILES[@]}"; do
    if [ -x "./$EXEC_FILE" ]; then
        echo "提交: ./$EXEC_FILE"
        RUN_TIME=$(date +"%H%M%S")
        LOG_FILE="$DATE_DIR/$(basename "$EXEC_FILE")_${RUN_TIME}.log"
        bsub -b -q "$QUEUE" -shared -n 1 -cgsp 64 -share_size 15000 \
             -o "$LOG_FILE" "./$EXEC_FILE"
        if [ $? -ne 0 ]; then
            echo "提交失败: ./$EXEC_FILE（请查看集群返回信息）"
        else
            echo "已提交: ./$EXEC_FILE（稍后查看 $LOG_FILE）"
        fi
    else
        echo "未找到可执行文件: ./$EXEC_FILE"
    fi
done
echo "=== Submission finished ==="
echo "所有任务已提交完成。日志保存在 $DATE_DIR"

