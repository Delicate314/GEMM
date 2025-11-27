#include <slave.h>
#include <simd.h>
#include <string.h>
#include <stdio.h>

static inline void print_top_left_block(
    const char* tag,
    int core_row, int core_col,
    const float* data, int rows, int cols, int ld) {
    printf("%s Core(%d,%d) rows=%d cols=%d\n",
        tag, core_row, core_col, rows, cols);
    for (int r = 0; r < rows; ++r) {
        printf("    ");
        for (int c = 0; c < cols; ++c) {
            printf("%f ", data[r * ld + c]);
        }
        printf("\n");
    }
}

#define kTileSize 64  // 每次沿 K 维度处理的列数
#define kSuperSize (8 * kTileSize)  // 每次处理 8 倍的 K 维度数据（256）

// 每核 C 分块最大尺寸（单位：行/列）
#define ASM_KMAXTILEM 64
#define ASM_KMAXTILEN 64

// --- LDM 内存布局 ---
// 每个从核都有 5 个 buffer
__thread_local float A_stage[ASM_KMAXTILEM * kTileSize] __attribute__((aligned(128)));  // DMA 缓冲区：每个从核 DMA 自己需要的 A 数据
__thread_local float B_stage[kTileSize * ASM_KMAXTILEN] __attribute__((aligned(128)));  // DMA 缓冲区：每个从核 DMA 自己需要的 B 数据
__thread_local float A_compute[ASM_KMAXTILEM * kTileSize] __attribute__((aligned(128)));  // 计算缓冲区：RMA 接收后的 A 数据
__thread_local float B_compute[kTileSize * ASM_KMAXTILEN] __attribute__((aligned(128)));  // 计算缓冲区：RMA 接收后的 B 数据
__thread_local float C_ldm[ASM_KMAXTILEM * ASM_KMAXTILEN] __attribute__((aligned(128)));  // 结果缓冲区


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
__thread_local athread_rply_t rma_l_rply_A;  // A 的本地回答字（非阻塞版需要）
__thread_local athread_rply_t rma_r_rply_A;  // A 的远程回答字
__thread_local athread_rply_t rma_l_rply_B;  // B 的本地回答字（非阻塞版需要）
__thread_local athread_rply_t rma_r_rply_B;  // B 的远程回答字

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
    // if (coreID_row == 0 && (coreID_col == 0 || coreID_col == CORE_GRID_DIM - 1)) {
    //     printf("[Debug] Core(%d,%d) my_tileM = %d\n", coreID_row, coreID_col, my_tileM);
    // }
    // if (coreID_col == 0 && (coreID_row == 0 || coreID_row == CORE_GRID_DIM - 1)) {
    //     printf("[Debug] Core(%d,%d) my_tileN = %d\n", coreID_row, coreID_col, my_tileN);
    // }

    // === 每核按最多 64x64 的块在行列方向分片（必要时处理尾块）===
    for (int tm = 0; tm < my_tileM; tm += ASM_KMAXTILEM) {
        // 计算当前子块的实际行数 curM：
        // - 如果 tm + ASM_KMAXTILEM <= my_tileM，说明剩余行数足够一个完整的 ASM_KMAXTILEM(64行)，则 curM = ASM_KMAXTILEM
        // - 否则，这是最后一个子块，剩余行数不足64行，则 curM = my_tileM - tm（剩余的行数）
        int curM = (tm + ASM_KMAXTILEM <= my_tileM) ? ASM_KMAXTILEM : (my_tileM - tm);
        for (int tn = 0; tn < my_tileN; tn += ASM_KMAXTILEN) {
            int curN = (tn + ASM_KMAXTILEN <= my_tileN) ? ASM_KMAXTILEN : (my_tileN - tn);
            // 清零当前子块 C（仅使用当前子块的实际尺寸）
            memset(C_ldm, 0, sizeof(float) * curM * curN);

            // 主循环：沿 K 维度按 kSuperSize (8 * kTileSize) 分块
            for (int k_block = 0; k_block < K; k_block += kSuperSize) {
                int curKSuper = (k_block + kSuperSize <= K) ? kSuperSize : (K - k_block);
                int numKTiles = (curKSuper + kTileSize - 1) / kTileSize;  // 实际需要处理的 kTileSize 块数（最多8个）

                // === 阶段1：每个从核 DMA 自己负责的 A / B 子块 ===
                // A 子块：按行划分（rid 决定 K 偏移），B 子块：按列划分（cid 决定 K 偏移）
                int k_offset_A = coreID_col * kTileSize;
                int k_offset_B = coreID_row * kTileSize;
                int curK_A = 0;
                int curK_B = 0;

                if (k_offset_A < curKSuper) {
                    curK_A = (k_offset_A + kTileSize <= curKSuper) ? kTileSize : (curKSuper - k_offset_A);
                    dma_rply_A = 0;
                    athread_dma_iget_stride(
                        A_stage,
                        &A_global[(row_start + tm) * K + k_block + k_offset_A],
                        curM * curK_A * sizeof(float),
                        curK_A * sizeof(float),
                        (K - curK_A) * sizeof(float),
                        &dma_rply_A);
                }

                if (k_offset_B < curKSuper) {
                    curK_B = (k_offset_B + kTileSize <= curKSuper) ? kTileSize : (curKSuper - k_offset_B);
                    dma_rply_B = 0;
                    athread_dma_iget_stride(
                        B_stage,
                        &B_global[(k_block + k_offset_B) * N + (col_start + tn)],
                        curK_B * curN * sizeof(float),
                        curN * sizeof(float),
                        (N - curN) * sizeof(float),
                        &dma_rply_B);
                }

                if (k_offset_A < curKSuper) {
                    athread_dma_wait_value(&dma_rply_A, 1);
                    // if (curK_A > 0 && k_offset_A == 0 &&
                    //     coreID_row == 0 && coreID_col == 0) {
                    //     print_top_left_block(
                    //         "[DMA A] top-left 5x5",
                    //         coreID_row, coreID_col,
                    //         A_stage, curM, curK_A, curK_A);
                    // }
                }
                if (k_offset_B < curKSuper) {
                    athread_dma_wait_value(&dma_rply_B, 1);
                    // if (curK_B > 0 && k_offset_B == 0 &&
                    //     coreID_row == 0 && coreID_col == 0) {
                    //     print_top_left_block(
                    //         "[DMA B] top-left 5x5",
                    //         coreID_row, coreID_col,
                    //         B_stage, curK_B, curN, curN);
                    // }
                }

                // === 阶段2：对 8 个 kTileSize 块进行循环，每次 RMA 广播并计算 ===
                for (int k = 0; k < numKTiles; ++k) {
                    int k_tile_offset = k * kTileSize;
                    int curK = (k_tile_offset + kTileSize <= curKSuper) ? kTileSize : (curKSuper - k_tile_offset);
                    rma_l_rply_A = 0;
                    rma_r_rply_A = 0;
                    athread_ssync(ROW_SCOPE, 0xff);
                    // A 的行广播：同一列的所有核（cid==k）各自在本行内广播自己 DMA 到的 A 数据
                    if (coreID_col == k) {
                        // if (coreID_row == 0) {
                        //     int print_rows = (curM < 5) ? curM : 5;
                        //     int print_cols = (curK < 5) ? curK : 5;
                        //     print_top_left_block(
                        //         "[RMA A] top-left 5x5",
                        //         coreID_row, coreID_col,
                        //         A_stage, print_rows, print_cols, curK);
                        // }
                        athread_rma_row_ibcast(
                            A_compute, A_stage,
                            curM * curK * sizeof(float),
                            &rma_l_rply_A,
                            &rma_r_rply_A);
                        athread_rma_wait_value(&rma_l_rply_A, 1);
                    }
                    athread_rma_wait_value(&rma_r_rply_A, 1);

                    // B 的列广播：同一行的所有核（rid==k）在列内广播自己 DMA 到的 B 数据
                    rma_l_rply_B = 0;
                    rma_r_rply_B = 0;
                    athread_ssync(COL_SCOPE, 0xff);
                    if (coreID_row == k) {
                        // if (coreID_col == 0) {
                        //     int print_rows = (curK < 5) ? curK : 5;
                        //     int print_cols = (curN < 5) ? curN : 5;
                        //     print_top_left_block(
                        //         "[RMA B] top-left 5x5",
                        //         coreID_row, coreID_col,
                        //         B_stage, print_rows, print_cols, curN);
                        // }
                        athread_rma_col_ibcast(
                            B_compute, B_stage,
                            curK * curN * sizeof(float),
                            &rma_l_rply_B,
                            &rma_r_rply_B);
                        athread_rma_wait_value(&rma_l_rply_B, 1);
                    }
                    athread_rma_wait_value(&rma_r_rply_B, 1);

                    // === 阶段3：SIMD 计算 ===
                    // 每个从核使用 A_compute 和 B_compute 进行计算
                    float* const __restrict__ current_C = C_ldm;
                    float* const __restrict__ current_A = A_compute;
                    float* const __restrict__ current_B = B_compute;
                    // SIMD 计算：使用 8x8 微内核（主路径），并对剩余行/列做回退处理
                    int aligned_M = (curM / 8) * 8;
                    int aligned_N = (curN / 8) * 8;

                    for (int i = 0; i < aligned_M; i += 8) {
                        for (int j = 0; j < aligned_N; j += 8) {
                            floatv8 c_sum_0, c_sum_1, c_sum_2, c_sum_3;
                            floatv8 c_sum_4, c_sum_5, c_sum_6, c_sum_7;
                            simd_load(c_sum_0, &current_C[(i + 0) * curN + j]);
                            simd_load(c_sum_1, &current_C[(i + 1) * curN + j]);
                            simd_load(c_sum_2, &current_C[(i + 2) * curN + j]);
                            simd_load(c_sum_3, &current_C[(i + 3) * curN + j]);
                            simd_load(c_sum_4, &current_C[(i + 4) * curN + j]);
                            simd_load(c_sum_5, &current_C[(i + 5) * curN + j]);
                            simd_load(c_sum_6, &current_C[(i + 6) * curN + j]);
                            simd_load(c_sum_7, &current_C[(i + 7) * curN + j]);

                            for (int k_inner = 0; k_inner < curK; ++k_inner) {
                                floatv8 b_vec;
                                simd_load(b_vec, &current_B[k_inner * curN + j]);

                                float a0 = current_A[(i + 0) * curK + k_inner];
                                float a1 = current_A[(i + 1) * curK + k_inner];
                                float a2 = current_A[(i + 2) * curK + k_inner];
                                float a3 = current_A[(i + 3) * curK + k_inner];
                                float a4 = current_A[(i + 4) * curK + k_inner];
                                float a5 = current_A[(i + 5) * curK + k_inner];
                                float a6 = current_A[(i + 6) * curK + k_inner];
                                float a7 = current_A[(i + 7) * curK + k_inner];

                                floatv8 a0_vec = simd_set_floatv8(a0, a0, a0, a0, a0, a0, a0, a0);
                                floatv8 a1_vec = simd_set_floatv8(a1, a1, a1, a1, a1, a1, a1, a1);
                                floatv8 a2_vec = simd_set_floatv8(a2, a2, a2, a2, a2, a2, a2, a2);
                                floatv8 a3_vec = simd_set_floatv8(a3, a3, a3, a3, a3, a3, a3, a3);
                                floatv8 a4_vec = simd_set_floatv8(a4, a4, a4, a4, a4, a4, a4, a4);
                                floatv8 a5_vec = simd_set_floatv8(a5, a5, a5, a5, a5, a5, a5, a5);
                                floatv8 a6_vec = simd_set_floatv8(a6, a6, a6, a6, a6, a6, a6, a6);
                                floatv8 a7_vec = simd_set_floatv8(a7, a7, a7, a7, a7, a7, a7, a7);

                                c_sum_0 = simd_vmas(a0_vec, b_vec, c_sum_0);
                                c_sum_1 = simd_vmas(a1_vec, b_vec, c_sum_1);
                                c_sum_2 = simd_vmas(a2_vec, b_vec, c_sum_2);
                                c_sum_3 = simd_vmas(a3_vec, b_vec, c_sum_3);
                                c_sum_4 = simd_vmas(a4_vec, b_vec, c_sum_4);
                                c_sum_5 = simd_vmas(a5_vec, b_vec, c_sum_5);
                                c_sum_6 = simd_vmas(a6_vec, b_vec, c_sum_6);
                                c_sum_7 = simd_vmas(a7_vec, b_vec, c_sum_7);
                            }

                            simd_store(c_sum_0, &current_C[(i + 0) * curN + j]);
                            simd_store(c_sum_1, &current_C[(i + 1) * curN + j]);
                            simd_store(c_sum_2, &current_C[(i + 2) * curN + j]);
                            simd_store(c_sum_3, &current_C[(i + 3) * curN + j]);
                            simd_store(c_sum_4, &current_C[(i + 4) * curN + j]);
                            simd_store(c_sum_5, &current_C[(i + 5) * curN + j]);
                            simd_store(c_sum_6, &current_C[(i + 6) * curN + j]);
                            simd_store(c_sum_7, &current_C[(i + 7) * curN + j]);
                        }

                        // 处理剩余列（不足 8 列）: 使用标量累积
                        for (int j = aligned_N; j < curN; ++j) {
                            for (int r = 0; r < 8; ++r) {
                                float acc = current_C[(i + r) * curN + j];
                                for (int k_inner = 0; k_inner < curK; ++k_inner) {
                                    acc += current_A[(i + r) * curK + k_inner] *
                                        current_B[k_inner * curN + j];
                                }
                                current_C[(i + r) * curN + j] = acc;
                            }
                        }
                    }

                    // 处理剩余行（不足 8 行），使用原有 4x8/标量策略
                    for (int i = aligned_M; i < curM; ++i) {
                        int j = 0;
                        // 向量处理列
                        for (; j + 7 < curN; j += 8) {
                            floatv8 c_sum;
                            simd_load(c_sum, &current_C[i * curN + j]);
                            for (int k_inner = 0; k_inner < curK; ++k_inner) {
                                floatv8 b_vec;
                                simd_load(b_vec, &current_B[k_inner * curN + j]);
                                float a_val = current_A[i * curK + k_inner];
                                floatv8 a_vec = simd_set_floatv8(a_val, a_val, a_val, a_val, a_val, a_val, a_val, a_val);
                                c_sum = simd_vmas(a_vec, b_vec, c_sum);
                            }
                            simd_store(c_sum, &current_C[i * curN + j]);
                        }
                        // 剩余列标量处理
                        for (; j < curN; ++j) {
                            float acc = current_C[i * curN + j];
                            for (int k_inner = 0; k_inner < curK; ++k_inner) {
                                acc += current_A[i * curK + k_inner] * current_B[k_inner * curN + j];
                            }
                            current_C[i * curN + j] = acc;
                        }
                    }
                }
            }

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