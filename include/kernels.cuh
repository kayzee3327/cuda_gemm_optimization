#ifndef KERNELS_H
#define KERNELS_H

// naive
__global__ void fp32gemm(float* A, float* B, float* C, 
    int M, int N, int K, float alpha, float beta);

__global__ void int32gemm(int* A, int* B, int* C, 
    int M, int N, int K, int alpha, int beta);

// coalesced
__global__ void coalesced_fp32gemm(
    float* A, float* B, float* C, 
    int M, int N, int K, 
    float alpha, float beta
);

// tile matrix multiplication using shared memory
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void smem_fp32gemm(
    float* A, float* B, float* C, 
    int M, int N, int K, 
    float alpha, float beta
) {
    __shared__ float tileA[TILE_M][TILE_K];
    __shared__ float tileB[TILE_K][TILE_N];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int transAPerThread = TILE_K / TILE_N; // TILE_M * TILE_K / (TILE_M * TILE_N)
    int transBPerThread = TILE_K / TILE_M; // similar

    float sum = 0.0;

    for (int i = 0; i < K / TILE_K; i++)
    {
        // transfer data
        // A
        for (int j = 0; j < transAPerThread; j++)
        {
            if (threadIdx.x + j * TILE_N < TILE_K)
            {
                tileA[threadIdx.y][threadIdx.x + j * TILE_N] = A[row * K + threadIdx.x + j * TILE_N + i * TILE_K];
            }
        }
        // B
        for (int j = 0; j < transBPerThread; j++)
        {
            if (threadIdx.y + j * TILE_M < TILE_K)
            {
                tileB[threadIdx.y + j * TILE_M][threadIdx.x] = B[(threadIdx.y + j * TILE_M + i * TILE_K) * N + col];
            }
        }

        __syncthreads();

        // compute
        for (int j = 0; j < TILE_K; j++)
        {
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
        __syncthreads();

    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

#endif