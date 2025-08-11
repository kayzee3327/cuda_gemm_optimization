#include <iostream>
#include <algorithm>

#include <stdlib.h>

#include "v2/gemm.cuh"

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
    
    if (bitsA == 32)
    {
        
        float *h_C = new float[M * N];

        std::fill_n(h_C, M * N, 0.0f);
        const int tb_width = 16;
        launchInnerProductSgemm(
            reinterpret_cast<float*>(h_A), 
            reinterpret_cast<float*>(h_B), 
            reinterpret_cast<float*>(h_C), 
            M, N, K,
            tb_width
        );

        // std::fill_n(h_C, M * N, 0.0f);
        // const int fragment_M = 256;
        // const int fragment_N = 128;
        // const int fragment_K = min(256, K);
        // launchOuterProductSgemm(
        //     reinterpret_cast<float*>(h_A), 
        //     reinterpret_cast<float*>(h_B), 
        //     reinterpret_cast<float*>(h_C),
        //     M, N, K,
        //     fragment_M, fragment_N, fragment_K
        // );

        std::fill_n(h_C, M * N, 0.0f);
        const int tM = 32;
        const int tN = 32;
        const int tK = 64;
        launchThreadblockInnerSgemm<tM, tN, tK>(
            reinterpret_cast<float*>(h_A), 
            reinterpret_cast<float*>(h_B), 
            reinterpret_cast<float*>(h_C),
            M, N, K
        );

        write_mat("resAB", M, N, 32, reinterpret_cast<uint8_t*>(h_C));
        delete[] h_C;
    }
    return 0;
}