#include "v1/kernels.cuh"

__global__ void coalesced_fp32gemm(
    float* A, float* B, float* C, 
    int M, int N, int K, 
    float alpha, float beta
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

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