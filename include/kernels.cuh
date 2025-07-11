#ifndef KERNELS_H
#define KERNELS_H

// naive
__global__ void fp32gemm(float* A, float* B, float* C, 
    int M, int N, int K, float alpha, float beta);

__global__ void int32gemm(int* A, int* B, int* C, 
    int M, int N, int K, int alpha, int beta);

// coalesced
__global__ void coalesced_fp32gemm(float* A, float* B, float* C, 
    int M, int N, int K, float alpha, float beta);

#endif