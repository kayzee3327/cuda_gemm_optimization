#include "v2/01_global_mem.cuh"

/**
 * @brief This function move data from global memory to register in a partially `coalesced` way, 
 * since A and B are both row-major stored
 * and then conduct matmul in a `inner-product` way
 * 
 * @param A M by K matrix, row-major stored
 * @param B K by N matrix, row-major stored
 * @param C M by N matrix, row-major stored
 */
__global__ void kInnerProductSgemm(
    float *A, float *B, float *C,
    int M, int N, int K,
    float alpha, float beta    
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float sum = 0.0;
        for (size_t i = 0; i < K; i++)
        {
            sum += A[row * K + i] * B[col + i * N];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void InnerProductSgemm::operator()(params& P) {
    kInnerProductSgemm<<<P.grid_shape, P.threadblock_shape>>>(
        P.d_A, P.d_B, P.d_C,
        P.M, P.N, P.K,
        P.alpha, P.beta
    );
}

/**
 * @brief This function move data from global memory to register in a partially `coalesced` way, 
 * since A and B are both row-major stored. Actually in this function, since there are double loops,
 * not many coalesced memory access can be achieved.
 * 
 * Then conduct matmul in a `outer-product` way, 
 * where we need to cut cols of A and rows of B into fragments to make the best of parallelism. 
 * This is similar to cutlass matmul in warp tile.
 * 
 * - However, performance will be downgraded by atomic operations in global memory.
 * 
 * - After exit the kernel, an `epilogue` with alpha and beta is required.
 * 
 * @param A M by K matrix, row-major stored
 * @param B K by N matrix, row-major stored
 * @param C M by N matrix, row-major stored
 */
__global__ void kOuterProductSgemm(
    float *A, float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    int fragment_M, 
    int fragment_N,
    int fragment_K
) {
    int offset_M = blockIdx.x * fragment_M;
    int offset_N = blockIdx.y * fragment_N;
    int offset_K = blockIdx.z * fragment_K;
    
    int idx_along_K = offset_K + threadIdx.x;
    int idx_along_M, idx_along_N;
    float accum;
    for (size_t i = 0; i < fragment_M; i++)
    {
        for (size_t j = 0; j < fragment_N; j++)
        {
            idx_along_M = offset_M + i;
            idx_along_N = offset_N + j;
            
            // this code will trigger a race condition, 
            // because multiple threads try to read and write to same location in `C`
            // `+=` is a read-and-write operation
            // C[idx_along_M * N + idx_along_N] += 
            //     A[idx_along_M * K + idx_along_K] * B[idx_along_K * N + idx_along_N];
            
            // use atomic add instead
            accum = A[idx_along_M * K + idx_along_K] * B[idx_along_K * N + idx_along_N];
            atomicAdd(&C[idx_along_M * N + idx_along_N], accum);
        }
    }
    // after exit the kernel, an epilogue with alpha and beta is required
}

void OuterProductSgemm::operator()(params& P) {
    kOuterProductSgemm<<<P.grid_shape, P.threadblock_shape>>>(
        P.d_A, P.d_B, P.d_C,
        P.M, P.N, P.K,
        P.alpha, P.beta,
        P.fragment_M, P.fragment_N, P.fragment_K
    );
}