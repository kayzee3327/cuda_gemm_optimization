#pragma once

#include "v2/params.cuh"

#include <cuda_runtime.h>

/**
 * @brief This function move data from global memory to register in a `coalesced` way, 
and then conduct matmul in a `inner-product` way
 * 
 * @param A M by K matrix, row-major stored
 * @param B K by N matrix, row-major stored
 * @param C M by N matrix, row-major stored
 */
class InnerProductSgemm {
    using params = Params<float>;
public:
    const char str[128] = "GMEM-InnerProductSgemm";

    void operator()(params& P);
};


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
class OuterProductSgemm {
    using params = Params<float>;
public:
    const char str[128] = "GMEM-OuterProductSgemm";

    void operator()(params& P);
};
