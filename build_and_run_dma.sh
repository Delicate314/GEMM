#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ================================
# 构建阶段
# ================================
echo "=== Building project ==="
swgcc -mslave -O3 -msimd -funroll-loops slave_dma.c -c -o slave.o #-funroll-loops
swgcc -faddress_align=128 -mhost -msimd -O3 master.c -c -o master.o
swg++ -faddress_align=128 -mhost -msimd -O3 main.cpp -c -o main.o
swg++ -mhybrid -static main.o master.o slave.o -o gemm_dma.exe
echo "=== Build finished ==="


# ================================
# 提交阶段
# ================================
QUEUE="q_sw_expr"
RESULT_DIR="./results/dma"
DATE_DIR="$RESULT_DIR/$(date +"%Y%m%d")"
mkdir -p "$DATE_DIR"

# 需要遍历的矩阵规模（M=N=K）
SIZES=(
    1536
    2048
    3072
    4096
    6144
    7680
    8192
    15360
)

EXEC_FILE="gemm_dma.exe"

echo "=== Submitting jobs ==="
if [ -x "./$EXEC_FILE" ]; then
    for SIZE in "${SIZES[@]}"; do
        M=$SIZE
        N=$SIZE
        K=$SIZE
        RUN_TIME=$(date +"%H%M%S")
        LOG_FILE="$DATE_DIR/$(basename "$EXEC_FILE")_${M}x${N}x${K}_${RUN_TIME}.log"
        echo "提交: ./$EXEC_FILE  (M=N=K=$SIZE)"
        bsub -b -q "$QUEUE" -shared -n 1 -cgsp 64 -share_size 15000 \
             -o "$LOG_FILE" "./$EXEC_FILE" "$M" "$N" "$K"
        if [ $? -ne 0 ]; then
            echo "  提交失败: ./$EXEC_FILE M=$M N=$N K=$K（请查看集群返回信息）"
        else
            echo "  已提交: ./$EXEC_FILE M=$M N=$N K=$K（日志：$LOG_FILE）"
        fi
    done
else
    echo "未找到可执行文件: ./$EXEC_FILE"
fi
echo "=== Submission finished ==="
echo "任务已提交完成。日志保存在 $DATE_DIR"

