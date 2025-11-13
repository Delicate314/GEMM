#include <slave.h>
#include <simd.h>
#include <string.h>

#define kTileSize 32

// 每核 C 分块最大尺寸（单位：行/列）
#define kMaxTileM 128
#define kMaxTileN 128

// --- LDM 内存布局 ---
// 1. 计算缓冲区 (所有核都有): 用于 SIMD 计算，需要双缓冲
__thread_local float A_compute[2][kMaxTileM * kTileSize] __attribute__((aligned(64)));
__thread_local float B_compute[2][kTileSize * kMaxTileN] __attribute__((aligned(64)));
__thread_local float C_ldm[kMaxTileM * kMaxTileN] __attribute__((aligned(64)));

// 2. DMA中转缓冲区: 用于暂存从DDR搬来的大块数据，也需要双缓冲
__thread_local float A_staging[2][kMaxTileM * kTileSize] __attribute__((aligned(64)));
__thread_local float B_staging[2][kTileSize * kMaxTileN] __attribute__((aligned(64)));


typedef struct {
    float* A; float* B; float* C;
    int M; int N; int K;
    int tileM; int tileN;
} GemmParams;

// --- DMA/RMA 回答字 ---
// DMA (DDR -> Leader LDM)
__thread_local volatile int dma_rply_A = 0;
__thread_local volatile int dma_rply_B = 0;

// RMA (Leader LDM -> Peer LDM)
__thread_local volatile int rma_l_rply_A = 0; // 发起核的本地回答字
__thread_local volatile int rma_r_rply_A = 0; // 接收核的远程回答字
__thread_local volatile int rma_l_rply_B = 0;
__thread_local volatile int rma_r_rply_B = 0;

void gemm(void* params) {
    GemmParams* gemmParams = (GemmParams*)params;
    float* const __restrict__ A_global = gemmParams->A;
    float* const __restrict__ B_global = gemmParams->B;
    float* const __restrict__ C_global = gemmParams->C;
    int M = gemmParams->M, N = gemmParams->N, K = gemmParams->K;
    int base_tileM = gemmParams->tileM, base_tileN = gemmParams->tileN;
    int coreID_row = _ROW, coreID_col = _COL;

    // --- 边界计算 ---
    const int CORE_GRID_DIM = 8;
    int remainder_M = M % CORE_GRID_DIM, remainder_N = N % CORE_GRID_DIM;
    // base_tileM/base_tileN 表示平均每核负责的行列块大小，无法整除时需要把余数分配给前 remainder_M/N 个核，每个多拿一行/列
    // row_start/col_start 计算方式：先按 base_tileM/N 计算基础偏移，再根据是否落在余数组内追加偏移以跳过前面核多出的那 1 行/列
    int row_start = coreID_row * base_tileM + (coreID_row < remainder_M ? coreID_row : remainder_M);
    int col_start = coreID_col * base_tileN + (coreID_col < remainder_N ? coreID_col : remainder_N);
    // my_tileM/my_tileN 计算方式：对落在余数段内的核多给 1 行/列，实现均匀分配
    int my_tileM = base_tileM + (coreID_row < remainder_M ? 1 : 0);
    int my_tileN = base_tileN + (coreID_col < remainder_N ? 1 : 0);

    // === 每核按最多 128x128 的块在行列方向分片（必要时处理尾块）===
    for (int tm = 0; tm < my_tileM; tm += kMaxTileM) {
        // 计算当前子块的实际行数 curM：
        // - 如果 tm + kMaxTileM <= my_tileM，说明剩余行数足够一个完整的 kMaxTileM(128行)，则 curM = kMaxTileM
        // - 否则，这是最后一个子块，剩余行数不足128行，则 curM = my_tileM - tm（剩余的行数）
        int curM = (tm + kMaxTileM <= my_tileM) ? kMaxTileM : (my_tileM - tm);
        for (int tn = 0; tn < my_tileN; tn += kMaxTileN) {
            int curN = (tn + kMaxTileN <= my_tileN) ? kMaxTileN : (my_tileN - tn);
            // 清零当前子块 C（仅使用当前子块的实际尺寸）
            memset(C_ldm, 0, sizeof(float) * curM * curN);

            // 初始化缓冲区索引（由 kSuper 循环内的 buf 控制 compute 双缓冲）

            // 主循环：沿 K 维度按 kSuper 条带处理；每核先 DMA 一次覆盖本轮，再进行 8 轮 RMA + 计算
            // kSuper = kTileSize * 8：每轮广播处理 kTileSize 个 K 元素，8 轮正好覆盖完整的条带
            const int kSuper = kTileSize * 8;
            for (int k_super = 0; k_super < K; k_super += kSuper) {
                // 每核 DMA 自己的 Aτ/Bτ（curM×kTileSize 与 kTileSize×curN），偏移与核坐标相关
                int a_k_off = k_super + (coreID_col % 8) * kTileSize;
                int b_k_off = k_super + (coreID_row % 8) * kTileSize;
                dma_rply_A = 0;
                athread_dma_iget_stride(
                    A_staging[0],
                    &A_global[(row_start + tm) * K + a_k_off],
                    curM * kTileSize * sizeof(float),
                    kTileSize * sizeof(float),
                    (K - kTileSize) * sizeof(float),
                    &dma_rply_A);
                dma_rply_B = 0;
                athread_dma_iget_stride(
                    B_staging[0],
                    &B_global[b_k_off * N + (col_start + tn)],
                    kTileSize * curN * sizeof(float),
                    curN * sizeof(float),
                    (N - curN) * sizeof(float),
                    &dma_rply_B);
                athread_dma_wait_value(&dma_rply_A, 1);
                athread_dma_wait_value(&dma_rply_B, 1);

                // 轮转 8 次：u=0..7
                for (int u = 0; u < 8; ++u) {
                    int buf = u % 2; // compute 双缓冲：偶数轮用0，奇数轮用1
                    // cid==u 广播 A
                    if (coreID_col == u) {
                        rma_l_rply_A = 0; rma_r_rply_A = 0;
                        athread_rma_row_ibcast(A_compute[buf], A_staging[0], curM * kTileSize * sizeof(float), &rma_l_rply_A, &rma_r_rply_A);
                    }
                    // rid==u 广播 B
                    if (coreID_row == u) {
                        rma_l_rply_B = 0; rma_r_rply_B = 0;
                        athread_rma_col_ibcast(B_compute[buf], &B_staging[0][0], curN * kTileSize * sizeof(float), &rma_l_rply_B, &rma_r_rply_B);
                    }
                    athread_ssync(ROW_SCOPE, 0xff); athread_rma_wait_value(&rma_r_rply_A, 1);
                    athread_ssync(COL_SCOPE, 0xff); athread_rma_wait_value(&rma_r_rply_B, 1);

                    // 计算 32
                    float* const __restrict__ current_A = A_compute[buf];
                    float* const __restrict__ current_B = B_compute[buf];
                    float* const __restrict__ current_C = C_ldm;
                    for (int i = 0; i < curM; i += 4) {
                        for (int j = 0; j < curN; j += 8) {
                            floatv8 c_sum_0, c_sum_1, c_sum_2, c_sum_3;
                            simd_load(c_sum_0, &current_C[(i + 0) * curN + j]);
                            simd_load(c_sum_1, &current_C[(i + 1) * curN + j]);
                            simd_load(c_sum_2, &current_C[(i + 2) * curN + j]);
                            simd_load(c_sum_3, &current_C[(i + 3) * curN + j]);
                            for (int k_inner = 0; k_inner < kTileSize; ++k_inner) {
                                floatv8 b_vec; simd_load(b_vec, &current_B[k_inner * curN + j]);
                                float a0 = current_A[i * kTileSize + k_inner];
                                float a1 = current_A[(i + 1) * kTileSize + k_inner];
                                float a2 = current_A[(i + 2) * kTileSize + k_inner];
                                float a3 = current_A[(i + 3) * kTileSize + k_inner];
                                c_sum_0 = simd_vmas(simd_set_floatv8(a0, a0, a0, a0, a0, a0, a0, a0), b_vec, c_sum_0);
                                c_sum_1 = simd_vmas(simd_set_floatv8(a1, a1, a1, a1, a1, a1, a1, a1), b_vec, c_sum_1);
                                c_sum_2 = simd_vmas(simd_set_floatv8(a2, a2, a2, a2, a2, a2, a2, a2), b_vec, c_sum_2);
                                c_sum_3 = simd_vmas(simd_set_floatv8(a3, a3, a3, a3, a3, a3, a3, a3), b_vec, c_sum_3);
                            }
                            simd_store(c_sum_0, &current_C[(i + 0) * curN + j]);
                            simd_store(c_sum_1, &current_C[(i + 1) * curN + j]);
                            simd_store(c_sum_2, &current_C[(i + 2) * curN + j]);
                            simd_store(c_sum_3, &current_C[(i + 3) * curN + j]);
                        }
                    }
                }
            }

            // （已由 kSuper 循环完成本子块的全部计算）

            // 写回当前子块
            volatile int dma_rply_C = 0;
            athread_dma_iput_stride(
                &C_global[(row_start + tm) * N + (col_start + tn)], C_ldm,
                curM * curN * sizeof(float),
                curN * sizeof(float),
                (N - curN) * sizeof(float),
                &dma_rply_C);
            athread_dma_wait_value(&dma_rply_C, 1);
        }
    }
}