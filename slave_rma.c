#include <slave.h>
#include <simd.h>
#include <string.h>

#define kTileSize 32  // 每次沿 K 维度处理的列数

// 每核 C 分块最大尺寸（单位：行/列）
#define kMaxTileM 64
#define kMaxTileN 64

// --- LDM 内存布局 ---
// 1. 计算缓冲区 (所有核都有): 用于 SIMD 计算，需要双缓冲
__thread_local float A_compute[2][kMaxTileM * kTileSize] __attribute__((aligned(64)));
__thread_local float B_compute[2][kTileSize * kMaxTileN] __attribute__((aligned(64)));
__thread_local float C_ldm[kMaxTileM * kMaxTileN] __attribute__((aligned(64)));

// 2. DMA 中转缓冲区：领导核用于从 DDR 搬运数据，再通过 RMA 广播给同行/列
__thread_local float A_stage[2][kMaxTileM * kTileSize] __attribute__((aligned(64)));
__thread_local float B_stage[2][kTileSize * kMaxTileN] __attribute__((aligned(64)));


typedef struct {
    float* A; float* B; float* C;
    int M; int N; int K;
    int tileM; int tileN;
} GemmParams;

// --- DMA/RMA 回答字 ---
// DMA (DDR -> Leader LDM)
__thread_local volatile int dma_rply_A_stage[2];
__thread_local volatile int dma_rply_B_stage[2];

// RMA (Leader LDM -> Peer LDM)
__thread_local volatile int rma_l_rply_A[2];
__thread_local volatile int rma_r_rply_A[2];
__thread_local volatile int rma_l_rply_B[2];
__thread_local volatile int rma_r_rply_B[2];

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
            for (int idx = 0; idx < curM * curN; ++idx) C_ldm[idx] = 0.0f;

            // --- K 方向流水线：DMA + RMA 双缓冲 ---
            int totalTiles = (K + kTileSize - 1) / kTileSize;
            if (totalTiles == 0) continue;

            int now = 0, next = 1;
            int tilesStarted = 0;   // 已经发起 DMA 的 K 面板数
            int tilesDone = 0;      // 已经完成计算的面板数
            int kBlock = 0;         // 下一次 DMA 的 K 维起始偏移

            int stageLen[2] = { 0, 0 };
            int computeLen[2] = { 0, 0 };

            // --- Step1 & Step2: 预取第一块到 staging[next] 并等待 ---
            if (tilesStarted < totalTiles) {
                int curK = (kBlock + kTileSize <= K) ? kTileSize : (K - kBlock);
                stageLen[next] = curK;
                if (coreID_col == 0) {
                    dma_rply_A_stage[next] = 0;
                    athread_dma_iget_stride(
                        A_stage[next],
                        &A_global[(row_start + tm) * K + kBlock],
                        curM * curK * sizeof(float),
                        curK * sizeof(float),
                        (K - curK) * sizeof(float),
                        &dma_rply_A_stage[next]);
                }
                if (coreID_row == 0) {
                    dma_rply_B_stage[next] = 0;
                    athread_dma_iget_stride(
                        B_stage[next],
                        &B_global[kBlock * N + (col_start + tn)],
                        curK * curN * sizeof(float),
                        curN * sizeof(float),
                        (N - curN) * sizeof(float),
                        &dma_rply_B_stage[next]);
                }
                tilesStarted++;
                kBlock += curK;
            }
            if (coreID_col == 0 && stageLen[next] > 0) athread_dma_wait_value(&dma_rply_A_stage[next], 1);
            if (coreID_row == 0 && stageLen[next] > 0) athread_dma_wait_value(&dma_rply_B_stage[next], 1);

            // --- Step3: 通过 RMA 将 staging[next] 分发到 compute[next] ---
            if (stageLen[next] > 0) {
                if (coreID_col == 0) {
                    rma_l_rply_A[next] = 0; rma_r_rply_A[next] = 0;
                    athread_rma_row_ibcast(A_compute[next], A_stage[next], curM * stageLen[next] * sizeof(float),
                        &rma_l_rply_A[next], &rma_r_rply_A[next]);
                }
                if (coreID_row == 0) {
                    rma_l_rply_B[next] = 0; rma_r_rply_B[next] = 0;
                    athread_rma_col_ibcast(B_compute[next], B_stage[next], stageLen[next] * curN * sizeof(float),
                        &rma_l_rply_B[next], &rma_r_rply_B[next]);
                }
                athread_ssync(ROW_SCOPE, 0xff);
                if (coreID_col == 0) athread_rma_wait_value(&rma_r_rply_A[next], 1);
                athread_ssync(COL_SCOPE, 0xff);
                if (coreID_row == 0) athread_rma_wait_value(&rma_r_rply_B[next], 1);
                computeLen[next] = stageLen[next];
                stageLen[next] = 0;
            }

            // --- Step4: 预取下一块到 staging[now]（若存在） ---
            if (tilesStarted < totalTiles) {
                int curK = (kBlock + kTileSize <= K) ? kTileSize : (K - kBlock);
                stageLen[now] = curK;
                if (coreID_col == 0) {
                    dma_rply_A_stage[now] = 0;
                    athread_dma_iget_stride(
                        A_stage[now],
                        &A_global[(row_start + tm) * K + kBlock],
                        curM * curK * sizeof(float),
                        curK * sizeof(float),
                        (K - curK) * sizeof(float),
                        &dma_rply_A_stage[now]);
                }
                if (coreID_row == 0) {
                    dma_rply_B_stage[now] = 0;
                    athread_dma_iget_stride(
                        B_stage[now],
                        &B_global[kBlock * N + (col_start + tn)],
                        curK * curN * sizeof(float),
                        curN * sizeof(float),
                        (N - curN) * sizeof(float),
                        &dma_rply_B_stage[now]);
                }
                tilesStarted++;
                kBlock += curK;
            }

            // --- Step5: 主循环 ---
            while (tilesDone < totalTiles) {
                // 5.0 切换 now/next
                int tmpBuf = now; now = next; next = tmpBuf;

                // 5.1 等待数据准备完成（此处已在广播函数中阻塞，补一个同步即可）
                athread_ssync(ROW_SCOPE, 0xff);
                athread_ssync(COL_SCOPE, 0xff);

                int curK = computeLen[now];
                float* const __restrict__ current_A = A_compute[now];
                float* const __restrict__ current_B = B_compute[now];
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
                tilesDone++;
                computeLen[now] = 0;

                if (tilesDone >= totalTiles) break;

                // 5.2 等待 DMA 完成并广播到 compute[next]
                if (stageLen[next] > 0) {
                    if (coreID_col == 0) athread_dma_wait_value(&dma_rply_A_stage[next], 1);
                    if (coreID_row == 0) athread_dma_wait_value(&dma_rply_B_stage[next], 1);

                    if (coreID_col == 0) {
                        rma_l_rply_A[next] = 0; rma_r_rply_A[next] = 0;
                        athread_rma_row_ibcast(A_compute[next], A_stage[next], curM * stageLen[next] * sizeof(float),
                            &rma_l_rply_A[next], &rma_r_rply_A[next]);
                    }
                    if (coreID_row == 0) {
                        rma_l_rply_B[next] = 0; rma_r_rply_B[next] = 0;
                        athread_rma_col_ibcast(B_compute[next], B_stage[next], stageLen[next] * curN * sizeof(float),
                            &rma_l_rply_B[next], &rma_r_rply_B[next]);
                    }
                    athread_ssync(ROW_SCOPE, 0xff);
                    if (coreID_col == 0) athread_rma_wait_value(&rma_r_rply_A[next], 1);
                    athread_ssync(COL_SCOPE, 0xff);
                    if (coreID_row == 0) athread_rma_wait_value(&rma_r_rply_B[next], 1);
                    computeLen[next] = stageLen[next];
                    stageLen[next] = 0;
                }

                // 5.36 发起下一块 DMA（若仍有剩余）
                if (tilesStarted < totalTiles) {
                    int curK_next = (kBlock + kTileSize <= K) ? kTileSize : (K - kBlock);
                    stageLen[now] = curK_next;
                    if (coreID_col == 0) {
                        dma_rply_A_stage[now] = 0;
                        athread_dma_iget_stride(
                            A_stage[now],
                            &A_global[(row_start + tm) * K + kBlock],
                            curM * curK_next * sizeof(float),
                            curK_next * sizeof(float),
                            (K - curK_next) * sizeof(float),
                            &dma_rply_A_stage[now]);
                    }
                    if (coreID_row == 0) {
                        dma_rply_B_stage[now] = 0;
                        athread_dma_iget_stride(
                            B_stage[now],
                            &B_global[kBlock * N + (col_start + tn)],
                            curK_next * curN * sizeof(float),
                            curN * sizeof(float),
                            (N - curN) * sizeof(float),
                            &dma_rply_B_stage[now]);
                    }
                    tilesStarted++;
                    kBlock += curK_next;
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