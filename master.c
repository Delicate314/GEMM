#include <athread.h>
typedef struct {
    float* A;
    float* B;
    float* C;
    int M;
    int N;
    int K;
    int tileM; // 这个现在代表 "基础块大小"
    int tileN; // 这个现在代表 "基础块大小"
} GemmParams;
extern SLAVE_FUN(gemm)(void*); //从核函数特有的声明方式
void GEMM(float* A, float* B, float* C, int M, int N, int K, int tileM, int tileN) {
    GemmParams params = {A, B, C, M, N, K, tileM, tileN};
    athread_spawn(gemm, &params); //启动从核
    athread_join();
}