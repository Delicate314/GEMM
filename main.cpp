#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>   // For fabs
#include <string>
#include <athread.h> // 主核接口

// 是否执行主核计算并验证结果
const bool VERIFY_RESULT = false;

// --- 从 master.c 引入的函数 ---
// 使用 extern "C" 来告诉 C++ 编译器这个函数是 C 风格的，防止名字修饰
extern "C" void GEMM(float* A, float* B, float* C, int M, int N, int K, int tileM, int tileN);


/**
 * @brief 初始化矩阵：编码行列信息（整数部分为行号，小数部分为列号/10000）
 * @param mat 指向矩阵内存的指针
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 */
void initialize_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            mat[idx] = static_cast<float>(i) * 0.001f + static_cast<float>(j) * 0.000001f;
            // mat[idx] =1.0f;
        }
    }
}

/**
 * @brief 在主核(CPU)上执行矩阵乘法，用于结果验证
 */
void gemm_cpu(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief 验证从核计算结果和主核计算结果是否一致
 * @return true 如果验证通过, false 如果失败
 */
bool verify(const float* C_slave, const float* C_cpu, int m, int n) {
    const float epsilon = 1e-4; // 允许的相对误差
    for (int i = 0; i < m * n; ++i) {
        if (fabs((C_slave[i] - C_cpu[i]) / C_cpu[i]) > epsilon && C_cpu[i] != 0) {
            std::cout << "Verification FAILED!" << std::endl;
            std::cout << "Mismatch at index " << i << " (row " << i / n << ", col " << i % n << ")" << std::endl;
            std::cout << "Slave result: " << C_slave[i] << ", CPU result: " << C_cpu[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification PASSED!" << std::endl;
    return true;
}


int main(int argc, char* argv[]) {
    int M = 2048;
    int N = 2048;
    int K = 2048;

    if (argc == 4) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }
    else if (argc == 1) {
        std::cout << "[Info] Using default size M=N=K=2048.\n";
    }
    else {
        std::cerr << "Usage: " << argv[0] << " [M N K]\n";
        return 1;
    }

    // 1. 初始化申威线程环境
    athread_init();

    // 2. 分配内存
    // 注意：为了与 C 风格的接口兼容，我们使用一维数组（指针）而不是 vector<vector>
    std::cout << "Allocating memory for matrices..." << std::endl;
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C_slave = new float[M * N]; // 用于存储从核计算结果

    // 3. 初始化矩阵
    std::cout << "Initializing matrices A and B (value = row*0.001 + col*0.000001)..." << std::endl;
    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);

    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Starting slave core computation for " << M << "x" << N << "x" << K << " GEMM..." << std::endl;

    // 4. 执行从核计算（含预热和多次测试）并计时
    const int CORE_GRID_DIM = 8;
    int base_tileM = M / CORE_GRID_DIM;
    int base_tileN = N / CORE_GRID_DIM;
    const int WARMUP_RUNS = 5;
    const int MEASURE_RUNS = 10;

    std::cout << "Warmup runs: " << WARMUP_RUNS << std::endl;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        GEMM(A, B, C_slave, M, N, K, base_tileM, base_tileN);
    }

    std::cout << "Measured runs: " << MEASURE_RUNS << std::endl;
    double total_seconds = 0.0;
    for (int run = 0; run < MEASURE_RUNS; ++run) {
        auto start_time = std::chrono::high_resolution_clock::now();
        GEMM(A, B, C_slave, M, N, K, base_tileM, base_tileN);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        total_seconds += elapsed.count();
    }

    double avg_seconds = total_seconds / MEASURE_RUNS;
    double gflops = (2.0 * M * N * K) / (avg_seconds * 1e9);

    std::cout << "Slave core computation finished." << std::endl;
    std::cout << "Average execution time: " << avg_seconds << " seconds" << std::endl;
    std::cout << "Performance (avg GFLOPS): " << gflops << std::endl;
    std::cout << "-------------------------------------------" << std::endl;


    // 6. (可选) 执行主核计算并验证结果
    if (VERIFY_RESULT) {
        std::cout << "Performing CPU computation for verification..." << std::endl;
        float* C_cpu = new float[M * N];

        auto cpu_start = std::chrono::high_resolution_clock::now();
        gemm_cpu(A, B, C_cpu, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;
        double cpu_seconds = cpu_elapsed.count();
        double cpu_gflops = (2.0 * M * N * K) / (cpu_seconds * 1e9);

        std::cout << "Verification:" << std::endl;
        bool ok = verify(C_slave, C_cpu, M, N);
        std::cout << "CPU time: " << cpu_seconds << " seconds" << std::endl;
        std::cout << "CPU GFLOPS: " << cpu_gflops << std::endl;
        std::cout << "Slave time (avg): " << avg_seconds << " seconds" << std::endl;
        std::cout << "Speedup (CPU/Slave): " << (cpu_seconds / avg_seconds) << std::endl;

        // 调试输出：打印左上角 4x4 的从核结果与CPU结果
        int dbg_rows = (M < 4 ? M : 4);
        int dbg_cols = (N < 4 ? N : 4);
        std::cout << "Top-left " << dbg_rows << "x" << dbg_cols << " block (Slave):\n";
        for (int i = 0; i < dbg_rows; ++i) {
            for (int j = 0; j < dbg_cols; ++j) {
                std::cout << C_slave[i * N + j] << (j + 1 == dbg_cols ? '\n' : ' ');
            }
        }
        std::cout << "Top-left " << dbg_rows << "x" << dbg_cols << " block (CPU):\n";
        for (int i = 0; i < dbg_rows; ++i) {
            for (int j = 0; j < dbg_cols; ++j) {
                std::cout << C_cpu[i * N + j] << (j + 1 == dbg_cols ? '\n' : ' ');
            }
        }

        int br_start_row = M - dbg_rows;
        int br_start_col = N - dbg_cols;
        std::cout << "Bottom-right " << dbg_rows << "x" << dbg_cols << " block (Slave):\n";
        for (int i = 0; i < dbg_rows; ++i) {
            for (int j = 0; j < dbg_cols; ++j) {
                int row = br_start_row + i;
                int col = br_start_col + j;
                std::cout << C_slave[row * N + col] << (j + 1 == dbg_cols ? '\n' : ' ');
            }
        }
        std::cout << "Bottom-right " << dbg_rows << "x" << dbg_cols << " block (CPU):\n";
        for (int i = 0; i < dbg_rows; ++i) {
            for (int j = 0; j < dbg_cols; ++j) {
                int row = br_start_row + i;
                int col = br_start_col + j;
                std::cout << C_cpu[row * N + col] << (j + 1 == dbg_cols ? '\n' : ' ');
            }
        }

        delete[] C_cpu;
    }

    // 7. 释放内存和申威线程环境
    delete[] A;
    delete[] B;
    delete[] C_slave;

    athread_halt();

    return 0;
}