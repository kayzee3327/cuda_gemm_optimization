#pragma once

#include <iostream>

#include "v2/params.cuh"

template<typename T, typename GemmKernel>
class Launch {
    using params = Params<T>;

private:
    cudaEvent_t start, stop;
    cudaStream_t stream = 0;
    float milliseconds = 0;

    GemmKernel kernel_wrapper;

public:
    Launch() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~Launch() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    /**
     * @brief run `GemmKernel` and output errors or running time
     * 
     * @param P kernel parameters
     */
    void run(params &P) {
        cudaMalloc((void**)&P.d_A, P.M * P.K * sizeof(float));
        cudaMalloc((void**)&P.d_B, P.K * P.N * sizeof(float));
        cudaMalloc((void**)&P.d_C, P.M * P.N * sizeof(float));

        cudaMemcpy(P.d_A, P.A, P.M * P.K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(P.d_B, P.B, P.K * P.N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(P.d_C, P.C, P.M * P.N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(start, stream);
        kernel_wrapper(P);
        cudaEventRecord(stop, stream);

        cudaEventSynchronize(stop);

        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            fprintf(stderr, "%s execution error: %s\n", kernel_wrapper.str, cudaGetErrorString(kernelErr));
            cudaFree(P.d_A);
            cudaFree(P.d_B);
            cudaFree(P.d_C);
            return;
        }

        cudaMemcpy(P.C, P.d_C, P.M * P.N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(P.d_A);
        cudaFree(P.d_B);
        cudaFree(P.d_C);
        // Calculate the elapsed time between the start and stop events
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("%s execution time: %f ms\n", kernel_wrapper.str, milliseconds);

    }

};
