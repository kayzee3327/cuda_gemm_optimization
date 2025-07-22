#ifndef KERNELS_H
#define KERNELS_H

#include <cmath>
#include <iostream>

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
// assert(tile_M == tile_N);
// assert(tile_K % tile_M == 0);
// assert(tile_K % tile_N == 0);
// assert(N % tile_N == 0);
// assert(M % tile_M == 0);
// assert(K % tile_K == 0);
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void smem_fp32gemm_with_constraints(
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

// assert(tile_K % tile_M == 0 or tile_M % tile_K == 0);
// assert(tile_K % tile_N == 0 or tile_N % tile_K == 0);
// assert(N % tile_N == 0);
// assert(M % tile_M == 0);
// assert(K % tile_K == 0);
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void smem_fp32gemm(
    float* A, float* B, float* C, 
    int M, int N, int K, 
    float alpha, float beta
) {
    __shared__ float tileA[TILE_M][TILE_K];
    __shared__ float tileB[TILE_K][TILE_N];

    int numTileA = TILE_M * TILE_K;
    int numTileB = TILE_K * TILE_N;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    int transPerThread = (int)(ceilf(
        (float)((TILE_M + TILE_N) * TILE_K) / 
        (float)(TILE_M * TILE_N)
    ));

    float sum = 0.0;

    for (int i = 0; i < K / TILE_K; i++)
    {
        // transfer
        for (int j = 0; j < transPerThread; j++)
        {
            int transId = threadId + j * TILE_M * TILE_N; // start from 0, indicating blockDim.y * blockDim.x == TILE_M * TILE_N
            int ty, tx;
            if (transId < numTileA)
            {
                ty = transId / TILE_K;
                tx = transId % TILE_K;
                
                tileA[ty][tx] = A[(blockIdx.y * TILE_M + ty) * K + i * TILE_K + tx]; // here indicates blockDim.y == TILE_M
            }
            else if (transId < numTileA + numTileB)
            {
                transId -= numTileA;
                ty = transId / TILE_N;
                tx = transId % TILE_N;

                tileB[ty][tx] = B[(i * TILE_K + ty) * N + blockIdx.x * TILE_N + tx];
            }
        }

        __syncthreads();
        sum = 0.0 + sum;
        
        // calculate
        for (int j = 0; j < TILE_K; j++)
        {
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
        __syncthreads();

    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
    

}

#endif