#include <athread.h>
#include "asm_bench.h"

extern SLAVE_FUN(asm_microkernel_bench)(void*);

void RUN_MICROKERNEL_BENCH(const AsmBenchParams* params) {
    AsmBenchParams local = *params;
    athread_spawn(asm_microkernel_bench, &local);
    athread_join();
}


