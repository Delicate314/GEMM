#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <athread.h>
#include "asm_bench.h"

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [repeats] [K] [core_freq_MHz]\n"
        << "  repeats        : micro-kernel repeat count (default 64)\n"
        << "  K              : inner dimension used in benchmark (<= 64, default 64)\n"
        << "  core_freq_MHz  : core frequency in MHz for GFLOPS conversion (default 1400)\n";
}

int main(int argc, char* argv[]) {
    AsmBenchParams params;
    params.repeats = 64;
    params.K = 64;
    params.core_freq_hz = 1.4e9;

    if (argc > 4) {
        print_usage(argv[0]);
        return 1;
    }
    if (argc >= 2) {
        params.repeats = std::max(1, std::atoi(argv[1]));
    }
    if (argc >= 3) {
        params.K = std::atoi(argv[2]);
        if (params.K <= 0 || params.K > 64) {
            std::cerr << "[Error] K must be in (0, 64].\n";
            return 1;
        }
    }
    if (argc == 4) {
        double freq_mhz = std::atof(argv[3]);
        if (freq_mhz <= 0) {
            std::cerr << "[Error] core_freq_MHz must be > 0.\n";
            return 1;
        }
        params.core_freq_hz = freq_mhz * 1e6;
    }

    std::cout << "[Host] Micro-kernel benchmark parameters:\n"
        << "  repeats = " << params.repeats << "\n"
        << "  K       = " << params.K << "\n"
        << "  freq    = " << params.core_freq_hz / 1e6 << " MHz\n";

    athread_init();
    RUN_MICROKERNEL_BENCH(&params);
    athread_halt();
    return 0;
}


