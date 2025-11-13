# 设置运行队列和通用参数
QUEUE="q_sw_expr"
COMMON_ARGS="--size=256 --output=./bench_result.csv"

# 生成的可执行文件目录
EXECUTABLE_DIR="."

# 定义需要运行的文件列表（相对路径）
EXEC_FILES=(
    "exe"
)

# 创建结果存放目录
RESULT_DIR="./results"
mkdir -p "$RESULT_DIR"

# 批量运行可执行文件
for EXEC_FILE in "${EXEC_FILES[@]}"; do
    FULL_PATH="$EXECUTABLE_DIR/$EXEC_FILE"

    if [ -x "$FULL_PATH" ]; then
        echo "正在运行: $FULL_PATH"

        # 提交到指定队列运行
        bsub -I -q "$QUEUE" "$FULL_PATH" $COMMON_ARGS > "$RESULT_DIR/$(basename "$EXEC_FILE").log" 2>&1

        if [ $? -ne 0 ]; then
            echo "运行失败: $FULL_PATH，已记录日志"
        else
            echo "运行完成: $FULL_PATH"
        fi
    else
        echo "未找到可执行文件: $FULL_PATH，跳过"
    fi
done

echo "所有任务已提交完成。日志保存在 $RESULT_DIR"