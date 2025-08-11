#pragma once

#include "v2/params.cuh"
#include "v2/iterator.cuh"

namespace SMEM
{

template<int tM, int tN, int tK>
/**
 * @brief This function move tiled data to shared memory (unified cache) first,
 * and then move data to register for calculation. Thus we can rearrange data
 * storage in SMEM instead of row-major way in GMEM and avoid bank conflicts.
 * 
 * 
 * @param A M by K matrix, row-major stored
 * @param B K by N matrix, row-major stored
 * @param C M by N matrix, row-major stored
 */
__global__ void kThreadblockInnerSgemm(
    float *A, float *B, float *C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float tileA[tM * tK]; // contiguous * strided
    __shared__ float tileB[tN * tK];

    const int k_threads = tM * tN;
    const int group_size_A = k_threads / tK;
    const int group_size_B = k_threads / tN;
    const int load_num_A = tM / group_size_A;
    const int load_num_B = tK / group_size_B;
    const int row = tM * blockIdx.y + threadIdx.x / tN;
    const int col = tN * blockIdx.x + threadIdx.x % tN;

    // row-major in gmem
    TileIterator<float> gmem_iter_A(
        A, K, M,
        0, tM * blockIdx.y,
        tK, tM * blockIdx.y + tM,
        threadIdx.x % tK, threadIdx.x / tK,
        tK
    );

    // col-major in smem
    TileIterator<float> smem_iter_A(
        tileA, tM, tK, 
        0, 0, tM, tK,
        threadIdx.x / tK, threadIdx.x % tK, 0
    );

    // row-major in gmem
    TileIterator<float> gmem_iter_B(
        B, N, K,
        tN * blockIdx.x, 0,
        tN * blockIdx.x + tN, tK,
        threadIdx.x % tN, threadIdx.x / tN,
        tK * N
    );

    // row-major in smem
    TileIterator<float> smem_iter_B(
        tileB, tN, tK,
        0, 0, tN, tK,
        threadIdx.x % tN, threadIdx.x / tN, 0
    );


    const int loopK = (K + tK - 1) / tK;
    
    float sum = 0.0;
    for (size_t k = 0; k < loopK; k++)
    {
        for (size_t a = 0; a < load_num_A; a++)
        {
            float fragment = gmem_iter_A.valid() ? gmem_iter_A.getCurrentElement() : 0;
            smem_iter_A.putElement(fragment);
            gmem_iter_A.next(0, group_size_A);
            smem_iter_A.next(group_size_A, 0);
        }
        for (size_t b = 0; b < load_num_B; b++)
        {
            float fragment = gmem_iter_B.valid() ? gmem_iter_B.getCurrentElement() : 0;
            smem_iter_B.putElement(fragment);
            gmem_iter_B.next(0, group_size_B);
            smem_iter_B.next(0, group_size_B);
        }
        gmem_iter_A.next_tile();
        gmem_iter_B.next_tile();
        smem_iter_A.reset();
        smem_iter_B.reset();
        __syncthreads();

        for (size_t c = 0; c < tK; c++)
        {
            sum += tileA[threadIdx.x / tN + c * tM] * tileB[threadIdx.x % tN + c * tN];
        }

        __syncthreads();
        
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

template<int tM, int tN, int tK>
class ThreadblockInnerSgemm {
    using params = Params<float>;
public:
    const char str[128] = "SMEM-ThreadblockInnerSgemm";

    void operator()(params& P) {
        kThreadblockInnerSgemm<tM, tN, tK><<<P.grid_shape, P.threadblock_shape>>>(
            P.d_A, P.d_B, P.d_C,
            P.M, P.N, P.K,
            P.alpha, P.beta
        );
    }
};



} // namespace SMEM
