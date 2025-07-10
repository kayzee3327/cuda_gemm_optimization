#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"

int main() {
    int rowsA, colsA, bitsA;
    int rowsB, colsB, bitsB;

    uint8_t *h_A, *h_B;

    read_mat("matrixA", rowsA, colsA, bitsA, h_A);
    read_mat("matrixB", rowsB, colsB, bitsB, h_B);

    if (colsA != rowsB) 
    {
        std::cerr << "colsA != rowsB" << std::endl;
        exit(1);
    }

    if (bitsA != bitsB)
    {
        std::cerr << "bitsA != bitsB" << std::endl;
        exit(1);
    }
    int M = rowsA, K = colsA, N = colsB;

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    if (bitsA == 32)
    {
        // Allocate device memory
        float *d_A, *d_B, *d_C, *h_C;
        h_C = new float[M * N];
        cudaMalloc((void**)&d_A, M * K * sizeof(float));
        cudaMalloc((void**)&d_B, K * N * sizeof(float));
        cudaMalloc((void**)&d_C, M * N * sizeof(float));
    
        // Copy data from host to device
        cublasSetVector(M * K, sizeof(float), reinterpret_cast<float*>(h_A), 1, d_A, 1);
        cublasSetVector(K * N, sizeof(float), reinterpret_cast<float*>(h_B), 1, d_B, 1);
    
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    
        // Set alpha and beta for C = alpha*A*B + beta*C
        float alpha = 1.0f;
        float beta = 0.0f;
    
        // Record start event
        cudaEventRecord(start);
    
        // Perform GEMM: C = A * B
        // Note the reversal of A and B for row-major to column-major conversion
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    
        // Record stop event
        cudaEventRecord(stop);
    
        // Wait for the GPU to finish
        cudaEventSynchronize(stop);
    
        // Calculate the elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "cuBLAS GEMM execution time: " << milliseconds << " ms" << std::endl;
    
        // Copy result from device to host
        cublasGetVector(M * N, sizeof(float), d_C, 1, h_C, 1);
        write_mat("resAB_cuBLAS", M, N, 32, reinterpret_cast<uint8_t*>(h_C));
    
        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cublasDestroy(handle);

        delete[] h_C;
    }
    delete[] h_A;
    delete[] h_B;

    return 0;
}