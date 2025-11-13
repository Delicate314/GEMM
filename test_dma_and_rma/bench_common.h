#pragma once

#include <stdint.h>

typedef struct {
    uint64_t dma_cycles;
    uint64_t row_send_cycles;
    uint64_t row_recv_cycles;
    uint64_t col_send_cycles;
    uint64_t col_recv_cycles;
} BenchRecord;

typedef struct {
    float* src;
    int elements;
    BenchRecord* records;
} BenchParams;

