#pragma once

#include "v2/01_global_mem.cuh"
#include "v2/02_shared_mem.cuh"
#include "v2/params.cuh"
#include "v2/launch.cuh"

#include <cuda_runtime.h>

void launchInnerProductSgemm(
    float *A, float *B, float *C,
    int M, int N, int K,
    const int tb_width
) {
    dim3 threadblock_shape(tb_width, tb_width, 1);
    dim3 grid_shape(
        (N + tb_width - 1) / tb_width,
        (M + tb_width - 1) / tb_width,
        1
    );

    Params<float> P(
        grid_shape,
        threadblock_shape,
        M, N, K,
        A, B, C,
        1.0, 0.0
    );
    
    Launch<float, GMEM::InnerProductSgemm> l;
    l.run(P);
}


void launchOuterProductSgemm(
    float *A, float *B, float *C,
    int M, int N, int K,
    const int fragment_M,
    const int fragment_N,
    const int fragment_K
) {
    dim3 threadblock_shape(fragment_K, 1, 1);
    dim3 grid_shape(
        (M + fragment_M - 1) / fragment_M,
        (N + fragment_N - 1) / fragment_N,
        (K + fragment_K - 1) / fragment_K
    );

    Params<float> P(
        grid_shape,
        threadblock_shape,
        M, N, K,
        A, B, C,
        1.0, 0.0
    );
    P.fragment_M = fragment_M;
    P.fragment_N = fragment_N;
    P.fragment_K = fragment_K;
    
    Launch<float, GMEM::OuterProductSgemm> l;
    l.run(P);
}

template<int tM, int tN, int tK>
void launchThreadblockInnerSgemm(
    float *A, float *B, float *C,
    int M, int N, int K
) {
    dim3 threadblock_shape(tN * tM, 1);
    dim3 grid_shape(
        (N + tN - 1) / tN,
        (M + tM - 1) / tM,
        1
    );

    Params<float> P(
        grid_shape,
        threadblock_shape,
        M, N, K,
        A, B, C,
        1.0, 0.0
    );
    
    Launch<float, SMEM::ThreadblockInnerSgemm<tM, tN, tK> > l;
    l.run(P);
}