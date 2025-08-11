#pragma once

#include <cuda_runtime.h>

template<typename T>
struct Params {

    // regular
    dim3 grid_shape;
    dim3 threadblock_shape;
    int M, N, K;
    T *A, *B, *C;
    T *d_A, *d_B, *d_C;
    T alpha, beta;

    // used in GMEM
    int fragment_M, fragment_N, fragment_K;

    Params(
        dim3 grid_shape,
        dim3 threadblock_shape,
        int M, int N, int K,
        T *A, T *B, T *C,
        T alpha, T beta
    ):
        grid_shape(grid_shape),
        threadblock_shape(threadblock_shape),
        M(M), N(N), K(K),
        A(A), B(B), C(C),
        alpha(alpha), beta(beta)
    {}
};