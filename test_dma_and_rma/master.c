#include <athread.h>

#include "bench_common.h"

extern SLAVE_FUN(bench_kernel)(void*);

void RUN_BENCHMARK(float* src, int elements, BenchRecord* records) {
    BenchParams params = { src, elements, records };
    athread_spawn(bench_kernel, &params);
    athread_join();
}

