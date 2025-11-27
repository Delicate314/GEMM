#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building micro-kernel benchmark ==="
swgcc -mslave -O3 -msimd -funroll-loops slave_asm.c -c -o slave_asm.o
swgcc -faddress_align=128 -mhost -msimd -O3 master_asm.c -c -o master_asm.o
swg++ -faddress_align=128 -mhost -msimd -O3 main_asm.cpp -c -o main_asm.o
swg++ -mhybrid -static main_asm.o master_asm.o slave_asm.o -o microkernel_bench.exe
echo "=== Build finished ==="

QUEUE="q_sw_expr"
RESULT_DIR="./results/asm"
DATE_DIR="$RESULT_DIR/$(date +"%Y%m%d")"
mkdir -p "$DATE_DIR"

REPEATS=${1:-64}
K_DIM=${2:-64}
CORE_FREQ_MHZ=${3:-1400}

echo "=== Submitting job ==="
RUN_TIME=$(date +"%H%M%S")
LOG_FILE="$DATE_DIR/microkernel_bench_${RUN_TIME}.log"

if [ ! -x "./microkernel_bench.exe" ]; then
    echo "Executable microkernel_bench.exe not found."
    exit 1
fi

bsub -b -q "$QUEUE" -shared -n 1 -cgsp 64 -share_size 15000 \
     -o "$LOG_FILE" "./microkernel_bench.exe" "$REPEATS" "$K_DIM" "$CORE_FREQ_MHZ"

echo "Submitted job. Log: $LOG_FILE"


