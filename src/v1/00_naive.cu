#include "v1/kernels.cuh"


__global__ void fp32gemm(
    float* A, float* B, float* C, 
    int M, int N, int K, 
    float alpha=1.0, float beta=0.0
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M and col < N)
    {
        float s = 0.0;
        for (int k = 0; k < K; k++)
        {
            s += A[row * K + k] * B[col + k * N];
        }
        C[row * N + col] = alpha * s + beta * C[row * N + col];
    }
}

__global__ void int32gemm(
    int* A, int* B, int* C, 
    int M, int N, int K, 
    int alpha, int beta
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M and col < N)
    {
        int s = 0;
        for (int k = 0; k < K; k++)
        {
            s += A[row * K + k] * B[col + k * N];
        }
        C[row * N + col] = alpha * s + beta * C[row * N + col];
    }
}
