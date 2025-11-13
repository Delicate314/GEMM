#include <slave.h>
#include <string.h>
#include <crts.h>
#include "bench_common.h"

static inline unsigned long long get_cycle() {
    unsigned long long t;
    asm volatile ("rcsr %0, 4" : "=r"(t));
    return t;
}

void bench_kernel(void* arg) {
    BenchParams* params = (BenchParams*)arg;
    int rid = _ROW;
    int cid = _COL;
    int idx = rid * 8 + cid;
    int isRowLeader = (rid == 0);
    int isColLeader = (cid == 0);
    int isCorner = (rid == 0 && cid == 0);

    int elements = params->elements;
    int bytes = elements * sizeof(float);

    BenchRecord record = { 0 };

    float* dmaBuf = (float*)ldm_malloc(bytes);
    float* rowDest = (float*)ldm_malloc(bytes);
    float* colDest = (float*)ldm_malloc(bytes);
    float* rowStage = NULL;
    float* colStage = NULL;

    if (isColLeader && bytes) {
        rowStage = (float*)ldm_malloc(bytes);
    }
    if (isRowLeader && bytes) {
        colStage = (float*)ldm_malloc(bytes);
    }

    if (bytes == 0 || dmaBuf == NULL || rowDest == NULL || colDest == NULL ||
        (isColLeader && rowStage == NULL) ||
        (isRowLeader && colStage == NULL)) {
        if (dmaBuf) ldm_free(dmaBuf, bytes);
        if (rowDest) ldm_free(rowDest, bytes);
        if (colDest) ldm_free(colDest, bytes);
        if (rowStage) ldm_free(rowStage, bytes);
        if (colStage) ldm_free(colStage, bytes);
        return;
    }

    // DMA benchmark (only corner leader performs)
    if (isCorner) {
        volatile int reply = 0;
        unsigned long long start = get_cycle();
        athread_dma_iget_stride(
            dmaBuf,
            params->src,
            bytes,
            bytes,
            0,
            &reply);
        athread_dma_wait_value(&reply, 1);
        unsigned long long end = get_cycle();
        record.dma_cycles = end - start;
    }
    // Row broadcast benchmark
    if (isRowLeader) {
        if (isColLeader) {
            // prepare stage buffer
            for (int i = 0; i < elements; ++i) {
                rowStage[i] = params->src[i];
            }
            volatile int lreply = 0;
            volatile int rreply = 0;
            unsigned long long start = get_cycle();
            athread_rma_row_ibcast(rowDest, rowStage, bytes, &lreply, &rreply);
            athread_ssync(ROW_SCOPE, 0xff);
            athread_rma_wait_value(&rreply, 1);
            unsigned long long end = get_cycle();
            record.row_send_cycles = end - start;
        }
        else {
            unsigned long long start = get_cycle();
            athread_ssync(ROW_SCOPE, 0xff);
            unsigned long long end = get_cycle();
            record.row_recv_cycles = end - start;
        }
    }
    // Column broadcast benchmark
    if (isColLeader) {
        if (isRowLeader) {
            for (int i = 0; i < elements; ++i) {
                colStage[i] = params->src[i];
            }
            volatile int lreply = 0;
            volatile int rreply = 0;
            unsigned long long start = get_cycle();
            athread_rma_col_ibcast(colDest, colStage, bytes, &lreply, &rreply);
            athread_ssync(COL_SCOPE, 0xff);
            athread_rma_wait_value(&rreply, 1);
            unsigned long long end = get_cycle();
            record.col_send_cycles = end - start;
        }
        else {
            unsigned long long start = get_cycle();
            athread_ssync(COL_SCOPE, 0xff);
            unsigned long long end = get_cycle();
            record.col_recv_cycles = end - start;
        }
    }
    params->records[idx] = record;

    ldm_free(dmaBuf, bytes);
    ldm_free(rowDest, bytes);
    ldm_free(colDest, bytes);
    if (rowStage) ldm_free(rowStage, bytes);
    if (colStage) ldm_free(colStage, bytes);
}

