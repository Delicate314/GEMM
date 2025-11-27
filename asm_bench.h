#ifndef ASM_BENCH_H_
#define ASM_BENCH_H_

typedef struct {
    int repeats;        // 微内核重复执行次数
    int K;              // K 维长度（<= MAX_BENCH_K）
    double core_freq_hz; // 从核主频（Hz）
} AsmBenchParams;

#ifdef __cplusplus
extern "C" {
#endif

    void RUN_MICROKERNEL_BENCH(const AsmBenchParams* params);

#ifdef __cplusplus
}
#endif

#endif  // ASM_BENCH_H_


