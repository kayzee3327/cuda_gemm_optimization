#include <iostream>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "utils.h"

#define NAIVE true
#define COALESCED true
#define SMEM true
#define THREAD1D false

// const int matrixA_rows = 8192;
// const int matrixA_cols = 2048;
// const int matrixB_rows = 2048;
// const int matrixB_cols = 4096;


// C = A @ B
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

    // Define the number of threads in each dimension of the thread block
    // A 16x16 block gives 256 threads, a good starting point.
    // const int TILE_WIDTH = 16;
    // dim3 threadPerBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Calculate the number of blocks needed in each dimension of the grid
    // This uses a ceiling division to ensure the grid covers the entire output matrix C (MxN)
    // dim3 numBlocks(
    //     (N + TILE_WIDTH - 1) / TILE_WIDTH,
    //     (M + TILE_WIDTH - 1) / TILE_WIDTH,
    //     1
    // );

    

    if (bitsA == 32)
    {
        float *h_C, alpha = 1.0, beta = 0.0;
        float *d_A, *d_B, *d_C;
        h_C = new float[M * N];

        cudaEvent_t start, stop;
        // Get the default stream
        cudaStream_t stream = 0;
        float milliseconds = 0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMalloc((void**)&d_A, M * K * sizeof(float));
        cudaMalloc((void**)&d_B, K * N * sizeof(float));
        cudaMalloc((void**)&d_C, M * N * sizeof(float));

        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

        if (NAIVE)
        {
            const int TILE_WIDTH = 8;
            dim3 threadPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 numBlocks(
                (M + TILE_WIDTH - 1) / TILE_WIDTH,
                (N + TILE_WIDTH - 1) / TILE_WIDTH,
                1
            );

            // Record the start event in the stream
            cudaEventRecord(start, stream);
            fp32gemm<<<numBlocks, threadPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            // Record the stop event in the stream
            cudaEventRecord(stop, stream);
            //Note: It's good practice to explicitly specify the stream (stream) to 
            //ensure the events and the kernel are in the same execution sequence.
            // Wait for the stop event to be recorded
            cudaError_t syncErr = cudaEventSynchronize(stop);
            if (syncErr != cudaSuccess) {
                fprintf(stderr, "Sync error: %s\n", cudaGetErrorString(syncErr));
            }
            cudaError_t kernelErr = cudaGetLastError();
            if (kernelErr != cudaSuccess) {
                fprintf(stderr, "naive execution error: %s\n", cudaGetErrorString(kernelErr));
            }
            // Calculate the elapsed time between the start and stop events
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("00_naive execution time: %f ms\n", milliseconds);
        }
        if (COALESCED)
        {
            const int TILE_WIDTH = 16;
            dim3 threadPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 numBlocks(
                (N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH,
                1
            );

            cudaEventRecord(start, stream);
            coalesced_fp32gemm<<<numBlocks, threadPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            cudaEventRecord(stop, stream);
            cudaError_t syncErr = cudaEventSynchronize(stop);
            if (syncErr != cudaSuccess) {
                fprintf(stderr, "Sync error: %s\n", cudaGetErrorString(syncErr));
            }cudaError_t kernelErr = cudaGetLastError();
            if (kernelErr != cudaSuccess) {
                fprintf(stderr, "01_coalesced execution error: %s\n", cudaGetErrorString(kernelErr));
            }
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("01_coalesced execution time: %f ms\n", milliseconds);
        }
        if (SMEM)
        {
            const int tile_M = 32;
            const int tile_N = 32;
            const int tile_K = 64;
            dim3 threadPerBlock(tile_M, tile_N, 1);
            dim3 numBlocks(
                (N + tile_N - 1) / tile_N,
                (M + tile_M - 1) / tile_M,
                1
            );
            // int shared_mem_size = (tile_M + tile_N) * tile_K * sizeof(float);
            
            // design assertion
            assert(tile_M == tile_N);
            assert(tile_K % tile_M == 0);
            assert(tile_K % tile_N == 0);
            assert(N % tile_N == 0);
            assert(M % tile_M == 0);
            assert(K % tile_K == 0);
            
            
            cudaEventRecord(start, stream);
            smem_fp32gemm<tile_M, tile_N, tile_K><<<numBlocks, threadPerBlock, 0>>>(
                d_A, d_B, d_C, M, N, K, alpha, beta
            );
            cudaEventRecord(stop, stream);
            cudaError_t syncErr = cudaEventSynchronize(stop);
            if (syncErr != cudaSuccess) {
                fprintf(stderr, "Sync error: %s\n", cudaGetErrorString(syncErr));
            }
            cudaError_t kernelErr = cudaGetLastError();
            if (kernelErr != cudaSuccess) {
                fprintf(stderr, "02_smem execution error: %s\n", cudaGetErrorString(kernelErr));
            }
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("02_smem execution time: %f ms\n", milliseconds);

        }
        if (THREAD1D)
        {
            /* code */
        }
        
        


        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        write_mat("resAB", M, N, 32, reinterpret_cast<uint8_t*>(h_C));
        delete[] h_C;
    }
    else if (bitsA == 0)
    {
        const int TILE_WIDTH = 16;
        dim3 threadPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 numBlocks(
            (N + TILE_WIDTH - 1) / TILE_WIDTH,
            (M + TILE_WIDTH - 1) / TILE_WIDTH,
            1
        );
        int *h_C, alpha = 1, beta = 0;
        int *d_A, *d_B, *d_C;
        h_C = new int[M * N];

        cudaEvent_t start, stop;
        // Get the default stream
        cudaStream_t stream = 0;
        float milliseconds = 0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMalloc((void**)&d_A, M * K * sizeof(int));
        cudaMalloc((void**)&d_B, K * N * sizeof(int));
        cudaMalloc((void**)&d_C, M * N * sizeof(int));

        cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, M * N * sizeof(int), cudaMemcpyHostToDevice);
        
        // Record the start event in the stream
        cudaEventRecord(start, stream);

        int32gemm<<<numBlocks, threadPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

        // Record the stop event in the stream
        cudaEventRecord(stop, stream);

        //Note: It's good practice to explicitly specify the stream (stream) to 
        //ensure the events and the kernel are in the same execution sequence.

        // Wait for the stop event to be recorded
        cudaError_t syncErr = cudaEventSynchronize(stop);
        if (syncErr != cudaSuccess) {
            fprintf(stderr, "Sync error: %s\n", cudaGetErrorString(syncErr));
        }

        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(kernelErr));
        }

        // Calculate the elapsed time between the start and stop events
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(h_C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        write_mat("resAB", M, N, 0, reinterpret_cast<uint8_t*>(h_C));
        delete[] h_C;

        printf("Kernel execution time: %f ms\n", milliseconds);
    }
    

    
    delete[] h_A;
    delete[] h_B;
    
    

    return 0;
}
