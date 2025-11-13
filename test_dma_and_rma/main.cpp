#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cstdint>
#include <athread.h>

#include "bench_common.h"

extern "C" void RUN_BENCHMARK(float* src, int elements, BenchRecord* records);

namespace {

    template <typename F>
    double average_cycles(const std::vector<BenchRecord>& records, F accessor) {
        std::uint64_t total = 0;
        std::uint64_t count = 0;
        for (const auto& rec : records) {
            std::uint64_t value = accessor(rec);
            if (value != 0) {
                total += value;
                ++count;
            }
        }
        return count ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
    }

} // namespace

int main() {
    athread_init();

    constexpr int elements = 1024;
    std::vector<float> src(elements);
    std::vector<BenchRecord> records(64);

    for (int i = 0; i < elements; ++i) {
        src[i] = static_cast<float>(i % 100) / 100.0f;
    }

    std::cout << "[Run] launching benchmark ...\n";
    RUN_BENCHMARK(src.data(), elements, records.data());
    std::cout << "[Run] benchmark finished.\n";

    auto avg_dma = average_cycles(records, [](const BenchRecord& r) { return r.dma_cycles; });
    auto avg_row_send = average_cycles(records, [](const BenchRecord& r) { return r.row_send_cycles; });
    auto avg_row_recv = average_cycles(records, [](const BenchRecord& r) { return r.row_recv_cycles; });
    auto avg_col_send = average_cycles(records, [](const BenchRecord& r) { return r.col_send_cycles; });
    auto avg_col_recv = average_cycles(records, [](const BenchRecord& r) { return r.col_recv_cycles; });

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[Result] average cycles:\n";
    std::cout << "  DMA (iget_stride): " << avg_dma << '\n';
    std::cout << "  Row broadcast send: " << avg_row_send << '\n';
    std::cout << "  Row broadcast recv: " << avg_row_recv << '\n';
    std::cout << "  Col broadcast send: " << avg_col_send << '\n';
    std::cout << "  Col broadcast recv: " << avg_col_recv << '\n';

    std::cout << "[Detail] per-core cycles (rid,cid,value):\n";
    for (int rid = 0; rid < 8; ++rid) {
        for (int cid = 0; cid < 8; ++cid) {
            const BenchRecord& rec = records[rid * 8 + cid];
            std::cout << "  (" << rid << "," << cid << ") DMA=" << rec.dma_cycles
                << " RowS=" << rec.row_send_cycles
                << " RowR=" << rec.row_recv_cycles
                << " ColS=" << rec.col_send_cycles
                << " ColR=" << rec.col_recv_cycles << '\n';
        }
    }

    athread_halt();
    return 0;
}