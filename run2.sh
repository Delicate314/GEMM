#!/bin/bash

# 设置运行队列
QUEUE="q_sw_expr"

# 创建结果存放目录
RESULT_DIR="./results"
DATE_DIR="$RESULT_DIR/$(date +"%Y%m%d")"
mkdir -p "$DATE_DIR"

# 定义需要运行的文件列表（相对路径）
EXEC_FILES=(
    "exe"
)

# 批量提交到指定队列运行（非交互）
for EXEC_FILE in "${EXEC_FILES[@]}"; do
    if [ -x "./$EXEC_FILE" ]; then
        echo "提交: ./$EXEC_FILE"
        RUN_TIME=$(date +"%H%M%S")
        LOG_FILE="$DATE_DIR/$(basename "$EXEC_FILE")_${RUN_TIME}.log"
        # 将作业标准输出写入 results/ 同名日志
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

echo "所有任务已提交完成。日志保存在 $DATE_DIR"