#include <slave.h>
#include <simd.h>
#include <string.h>

#define kTileSize 32  // 每次沿 K 维度处理的列数

// 每核 C 分块最大尺寸（单位：行/列）
#define kMaxTileM 64
#define kMaxTileN 64

// --- LDM 内存布局 ---
// 1. 计算缓冲区 (所有核都有): 用于 SIMD 计算
__thread_local float A_panel[kMaxTileM * kTileSize] __attribute__((aligned(64)));
__thread_local float B_panel[kTileSize * kMaxTileN] __attribute__((aligned(64)));
__thread_local float C_ldm[kMaxTileM * kMaxTileN] __attribute__((aligned(64)));


typedef struct {
    float* A; float* B; float* C;
    int M; int N; int K;
    int tileM; int tileN;
} GemmParams;

// --- DMA/RMA 回答字 ---
// DMA (DDR -> Leader LDM)
__thread_local volatile int dma_rply_A = 0;
__thread_local volatile int dma_rply_B = 0;

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

            // 主循环：沿 K 维度按固定宽度（kTileSize）分块，仅使用 DMA 搬运
            for (int k_block = 0; k_block < K; k_block += kTileSize) {
                int curK = (k_block + kTileSize <= K) ? kTileSize : (K - k_block);

                dma_rply_A = 0;
                athread_dma_iget_stride(
                    A_panel,
                    &A_global[(row_start + tm) * K + k_block],
                    curM * curK * sizeof(float),
                    curK * sizeof(float),
                    (K - curK) * sizeof(float),
                    &dma_rply_A);
                dma_rply_B = 0;
                athread_dma_iget_stride(
                    B_panel,
                    &B_global[k_block * N + (col_start + tn)],
                    curK * curN * sizeof(float),
                    curN * sizeof(float),
                    (N - curN) * sizeof(float),
                    &dma_rply_B);
                athread_dma_wait_value(&dma_rply_A, 1);
                athread_dma_wait_value(&dma_rply_B, 1);

                float* const __restrict__ current_A = A_panel;
                float* const __restrict__ current_B = B_panel;
                float* const __restrict__ current_C = C_ldm;
                for (int i = 0; i < curM; i += 4) {
                    for (int j = 0; j < curN; j += 8) {
                        floatv8 c_sum_0, c_sum_1, c_sum_2, c_sum_3;
                        simd_load(c_sum_0, &current_C[(i + 0) * curN + j]);
                        simd_load(c_sum_1, &current_C[(i + 1) * curN + j]);
                        simd_load(c_sum_2, &current_C[(i + 2) * curN + j]);
                        simd_load(c_sum_3, &current_C[(i + 3) * curN + j]);
                        for (int k_inner = 0; k_inner < curK; ++k_inner) {
                            floatv8 b_vec; simd_load(b_vec, &current_B[k_inner * curN + j]);
                            float a0 = current_A[(i + 0) * curK + k_inner];
                            float a1 = current_A[(i + 1) * curK + k_inner];
                            float a2 = current_A[(i + 2) * curK + k_inner];
                            float a3 = current_A[(i + 3) * curK + k_inner];
                            floatv8 a0_vec = simd_set_floatv8(a0, a0, a0, a0, a0, a0, a0, a0);
                            floatv8 a1_vec = simd_set_floatv8(a1, a1, a1, a1, a1, a1, a1, a1);
                            floatv8 a2_vec = simd_set_floatv8(a2, a2, a2, a2, a2, a2, a2, a2);
                            floatv8 a3_vec = simd_set_floatv8(a3, a3, a3, a3, a3, a3, a3, a3);
                            c_sum_0 = simd_vmas(a0_vec, b_vec, c_sum_0);
                            c_sum_1 = simd_vmas(a1_vec, b_vec, c_sum_1);
                            c_sum_2 = simd_vmas(a2_vec, b_vec, c_sum_2);
                            c_sum_3 = simd_vmas(a3_vec, b_vec, c_sum_3);
                        }
                        simd_store(c_sum_0, &current_C[(i + 0) * curN + j]);
                        simd_store(c_sum_1, &current_C[(i + 1) * curN + j]);
                        simd_store(c_sum_2, &current_C[(i + 2) * curN + j]);
                        simd_store(c_sum_3, &current_C[(i + 3) * curN + j]);
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