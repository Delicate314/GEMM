#include <slave.h>
#include <simd.h>
#include <crts.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "asm_bench.h"

#define MAX_BENCH_BLOCK_M 16
#define MAX_BENCH_BLOCK_N 16
#define MAX_BENCH_K 64
#define MAX_VEC_COLS (MAX_BENCH_BLOCK_N / 8)

#define DEFAULT_BENCH_REPEATS 64
#define DEFAULT_CORE_FREQ_HZ 1400000000.0  // 默认 1.4GHz，可按需修改

typedef struct {
    int blockM;
    int blockN;
} MicroKernelConfig;

typedef struct {
    int blockM;
    int blockN;
    uint64_t cycles;
    double gflops;
} MicroKernelResult;

static const MicroKernelConfig kConfigs[] = {
    {4, 8},
    {8, 8},
    {8, 16},
    {12, 8},
    {16, 8},
    {16, 16}
};

static __thread_local float A_tile[MAX_BENCH_BLOCK_M * MAX_BENCH_K] __attribute__((aligned(64)));
static __thread_local float B_tile[MAX_BENCH_K * MAX_BENCH_BLOCK_N] __attribute__((aligned(64)));
static __thread_local float C_tile[MAX_BENCH_BLOCK_M * MAX_BENCH_BLOCK_N] __attribute__((aligned(64)));

static inline uint64_t get_cycle() {
    uint64_t value;
    asm volatile("rcsr %0, 4" : "=r"(value));
    return value;
}

static void init_benchmark_data() {
    for (int i = 0; i < MAX_BENCH_BLOCK_M; ++i) {
        for (int k = 0; k < MAX_BENCH_K; ++k) {
            A_tile[i * MAX_BENCH_K + k] = (float)(0.01 * i + 0.001 * k);
        }
    }
    for (int k = 0; k < MAX_BENCH_K; ++k) {
        for (int j = 0; j < MAX_BENCH_BLOCK_N; ++j) {
            B_tile[k * MAX_BENCH_BLOCK_N + j] = (float)(0.02 * k + 0.0005 * j);
        }
    }
    memset(C_tile, 0, sizeof(C_tile));
}

static void run_kernel_once(int blockM, int blockN, int K) {
    const int vecCols = blockN / 8;
    floatv8 c_acc[MAX_BENCH_BLOCK_M][MAX_VEC_COLS];

    // 载入初始 C
    for (int r = 0; r < blockM; ++r) {
        for (int c = 0; c < vecCols; ++c) {
            simd_load(c_acc[r][c], &C_tile[r * MAX_BENCH_BLOCK_N + c * 8]);
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < vecCols; ++c) {
            floatv8 b_vec;
            simd_load(b_vec, &B_tile[k * MAX_BENCH_BLOCK_N + c * 8]);
            for (int r = 0; r < blockM; ++r) {
                float a_val = A_tile[r * MAX_BENCH_K + k];
                floatv8 a_vec = simd_set_floatv8(
                    a_val, a_val, a_val, a_val, a_val, a_val, a_val, a_val);
                c_acc[r][c] = simd_vmas(a_vec, b_vec, c_acc[r][c]);
            }
        }
    }

    for (int r = 0; r < blockM; ++r) {
        for (int c = 0; c < vecCols; ++c) {
            simd_store(c_acc[r][c], &C_tile[r * MAX_BENCH_BLOCK_N + c * 8]);
        }
    }
}

static uint64_t run_microkernel(int blockM, int blockN, int K, int repeats) {
    uint64_t start = get_cycle();
    for (int iter = 0; iter < repeats; ++iter) {
        run_kernel_once(blockM, blockN, K);
    }
    uint64_t end = get_cycle();
    return end - start;
}

static int is_valid_config(int blockM, int blockN) {
    if (blockM <= 0 || blockN <= 0) return 0;
    if (blockM > MAX_BENCH_BLOCK_M || blockN > MAX_BENCH_BLOCK_N) return 0;
    if (blockN % 8 != 0) return 0;
    return 1;
}

void asm_microkernel_bench(void* params) {
    AsmBenchParams defaults = {
        DEFAULT_BENCH_REPEATS,
        MAX_BENCH_K,
        DEFAULT_CORE_FREQ_HZ
    };
    AsmBenchParams cfg = defaults;
    if (params != NULL) {
        AsmBenchParams* user = (AsmBenchParams*)params;
        if (user->repeats > 0) cfg.repeats = user->repeats;
        if (user->K > 0 && user->K <= MAX_BENCH_K) cfg.K = user->K;
        if (user->core_freq_hz > 0.0) cfg.core_freq_hz = user->core_freq_hz;
    }

    init_benchmark_data();

    const int configCount = sizeof(kConfigs) / sizeof(kConfigs[0]);
    int coreID_row = _ROW;
    int coreID_col = _COL;
    int is_master_core = (coreID_row == 0 && coreID_col == 0);

    MicroKernelResult best = { 0, 0, UINT64_MAX, 0.0 };

    if (is_master_core) {
        printf("[MicroKernel] Benchmark start (repeats=%d, K=%d, freq=%.2f MHz)\n",
            cfg.repeats, cfg.K, cfg.core_freq_hz / 1e6);
    }
    for (int idx = 0; idx < configCount; ++idx) {
        int blockM = kConfigs[idx].blockM;
        int blockN = kConfigs[idx].blockN;
        if (!is_valid_config(blockM, blockN)) {
            if (is_master_core) {
                printf("  Skip invalid config %dx%d\n", blockM, blockN);
            }
            continue;
        }

        memset(C_tile, 0, sizeof(C_tile));
        uint64_t cycles = run_microkernel(blockM, blockN, cfg.K, cfg.repeats);
        double flops = 2.0 * blockM * blockN * cfg.K * cfg.repeats;
        double seconds = cycles / cfg.core_freq_hz;
        double gflops = (seconds > 0.0) ? (flops / seconds / 1e9) : 0.0;

        if (is_master_core) {
            printf("  %2dx%-2d : cycles=%llu, GFLOPS=%.2f\n",
                blockM, blockN, (unsigned long long)cycles, gflops);
        }

        // 按 GFLOPS 选择最佳配置（吞吐优先）
        if (gflops > best.gflops) {
            best.blockM = blockM;
            best.blockN = blockN;
            best.cycles = cycles;
            best.gflops = gflops;
        }
    }

    if (is_master_core) {
        if (best.cycles == UINT64_MAX) {
            printf("[MicroKernel] No valid configuration tested.\n");
        }
        else {
            printf("[MicroKernel] Best config (by GFLOPS): %dx%d (cycles=%llu, GFLOPS=%.2f)\n",
                best.blockM, best.blockN,
                (unsigned long long)best.cycles,
                best.gflops);
        }
    }
}

