# cublas baseline: 29.6072 ms

# 悬而未决

1. 当block size是16\*16时，计算速度最快，但是当block size是8\*8时，coalesced版本比naive版本慢，反而是naive版本达到了300ms左右的性能，为什么？（此时grid的M与N搞反了）
    
    而当M与N恢复正常时，block size是8\*8时，naive的速度变快了，coalesced的速度变慢了
    
    - naive访问如果需要速度提高，需要尽可能用到并行thread block数，而coalesced需要保证同时取的数据够多

2. WSL上做profiling有一些[问题](https://forums.developer.nvidia.com/t/nsight-system-error-unified-memory-cannot-be-traced-on-devices-that-dont-support-peer-to-peer-transfers/294742)，目前只能转移到Windows上编译运行并profile文件

3. 当grid大小的M与N交换后，为什么即使blockIdx对应不上，结果也能正确而且速度快上一倍？

4. 如何在nsight compute查看bank conflict

# cuda写法

首先要理解内置参数`blockIdx`、`blockDim`、`threadIdx`，理解它们和n卡体系结构的联系。
https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html

## 体系结构与编程模型

体系结构可以简单分为以下层次：

1. high level

    ![](pics/simple-gpu-arch.svg)

    - Streaming Multiprocessors (SMs)

    - on-chip L2 cache

    - high-bandwidth DRAM

    DRAM的数据访问都要经过L2 Cache

2. SMs

    - its own instruction schedulers 

    - various instruction execution pipelines

    根据不同的架构，operations可以在tensor core或cuda core上执行

编程模型中把threads组织在block，grid两层结构中，

- threads被均匀分配在等大的block中

- 当GPU执行时，有些instructions依赖于其他instructions的结果，需要等待，GPU会执行其他thread来减少空闲时间。因此，用户需要设置比core或instruction pipeline多得多的thread数

- 执行时，一个block会被完整的放在一个SM上，来保证通信效率

- 一个SM可以并行执行很多个block，所以一般设置block数量是SM数量的好几倍，这么做是为了避免"tail effect"，如图

    > Figure 3. Utilization of an 8-SM GPU when 12 thread blocks with an occupancy of 1 block/SM at a time are launched for execution. Here, the blocks execute in 2 waves, the first wave utilizes 100% of the GPU, while the 2nd wave utilizes only 50%.
    ![](pics/utilize-8sm-gpu.svg)

- 在真实的SM中，thread会分配给warp中的worker，一般一个warp32个worker

## 运用内置变量

cuda提供了2D或3D的方式来设置block和grid，分别用来并行2D和3D数据，比如处理图片用2D，物理三维模拟用3D，这样做可以形象地把thread block和数据块对应起来，或者说把thread block中的thread和数据块中的一个数据点，以平面或立体的方式对应起来。可以方便的计算出在原数据中的相应访问位置，比如矩阵的col number和row number，如下

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

# 实现简单的cuda矩阵乘: 579.966797 ms

首先将`C[i, j]`并行到每个thread

### 整理项目

将文件分别存放在`include` `data` `build` `bin` `src`等文件夹中

### 矩阵数据生成

使用二进制`.bin`文件按位存放，parsing更快，使用空间更小，但要注意大小端

### cuda kernel运行时间计算

首先用`cudaEventRecord(start, stream);`，`cudaEventRecord(stop, stream);`精确地记录kernel开始结束时间

- 显式使用参数`stream`，此处为0即默认`stream`，可以保证`start`和`stop`事件与kernel在同一执行序列中

- 在kernel前后记录事件，保证了kernel开始结束都被精确记录

然后再做`cudaEventSynchronize`，

然后再获取错误信息`cudaGetLastError`，

如果没有问题，就可以用`cudaEventElapsedTime`计算时间，最后要用`cudaEventDestroy`来销毁事件

### warning #549-D: variable "h_A" is used before its value is set

`void read_mat(std::string matname, int& r, int& c, int& b, uint8_t*& matret);`

这里函数`read_mat`因为要根据是哪种fp来申请对应的空间，需要在函数内对传入指针`matret`通过`new`动态申请空间，所以不能传递指针的副本，而是需要指针的引用。

### numpy生成矩阵

numpy的randint会使用int64作为默认dtype

# 从non-coalesced memory access到coalesced memory access: 279.250946 ms

在gpu上，一个线程块会被划分到多个warp上来执行，一般一个warp上有32个worker，那么各个warp会按顺序取0-31，32-63等等编号的线程。在2D的线程块中，线程编号是这样计算的`threadID = threadIdx.x + threadIdx.y * blockDim.x`，这意味着warp在线程块中会按行取线程，即row-major。

按照naive矩阵乘的方法，每个线程中

```cpp
row = blockIdx.x * blockDim.x + threadIdx.x;
col = blockIdx.y * blockDim.y + threadIdx.y;
```

横x竖y符合直觉。但是在计算矩阵时如下读AB矩阵、写C矩阵操作中

```cpp
sum += A[row * K + k] * B[col + k * N];
C[row * N + col] = alpha * s + beta * C[row * N + col];
```

可以发现A和C矩阵的访问都涉及到了`row*行宽+col`，说明一个warp中连续编号的threads，在此处读写时都要跨行或者说按列取，而不能实现连续的threads取连续的内存。

- 此处可能会误解连续取内存是每个线程中k循环算一个sum，在cuda的simt背景下，让并行的threads取连续的内存效率更高，更重要。

所以尝试把访问内存并行起来

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

为什么这样改呢？交换x和y之后，一个warp中的线程是对应原来的列上的，而`row*行宽+col`对于`col`是连续的，这样子一个warp可以并行地取连续的内存，大大提高内存访问效率。

### `float const* p` `const float* p` `float* const p`

前两个的`const`对应的是`float`，意思都是`p`指向的值不能变。

最后一个的`const`对应的是`p`，意思是`p`的指向不能变，但是指向的值可以变


# 利用shared memory：2D block tiling：188.578812 ms

此时的程序需要在global memory大量读写，每计算一个C的entry，就要从global memory中取出一行一列，不光访问速度效率最低，而且其中还会取到重复的行列，效率不高。于是考虑利用位于每个SM上的shared memory，以增加数据复用效率。

首先要知道SMEM的特征：

- 每个thread block独享自己的一份SMEM slice，只能由这个thread block访问修改

从最简单的想法出发，在一个SM的SMEM存一行的数据，让一个thread block重复用这一行去计算结果，但是会遇到很大的矩阵，比如现在的`8192-2048 * 2048-4096`，会有这些问题：

- 矩阵一行一列需要的空间比较大，接近或超过SMEM的大小

- 一个thread block的线程数量上限是1024，没办法处理这么多entry

所以需要利用好thread数量和SMEM的上限，我们可以考虑用tiled matrix multiplication，这样就可以灵活的调整放在SMEM的数据和并行线程了。有如下限制需要考虑：

1. 一是架构，我的卡是compute capability 8.6，可以得知每个SM上

    - 可以同时并行16个thread blocks

    - 同时可以运行48个warp，每个32个workers

    - unified data cache（包括L1 data cache和shared memory）是128kB，可以划分给shared memory 0, 8, 16, 32, 64 or 100 kB的空间，即shared memory最大是100kB

    - 一个thread block最多使用99kB shared memory

2. 二是SMEM和thread block的限制

    - thread block与grid尺寸设计一般与矩阵形状绑定
    
    - tile尺寸越小，global memory和SMEM之间的数据迁移就越多，数据复用减少

    - thread block越小，即`TILE_M`和`TILE_N`越小，如果增加`TILE_K`来维持数据规模，thread少会导致数据迁移时间增多。

    - 一个thread block负责的数据太多，即`TILE_M`\*`TILE_N`\*`TILE_K`很大，那么SMEM的尺寸限制会使后续thread block难以运行，降低了thread blocks之间的并行程度

那么为了找到最合适的`TILE_M`、`TILE_N`、`TILE_K`，需要从相对大的尺寸入手，既保证数据迁移效率又能不损失太多并行度，然后逐渐调优。因为需要均分数据迁移工作，暂时设计ratio \* `TILE_M` = ratio \* `TILE_N` = `TILE_K`

### 选择thread block dim
    
thread block有两种选择：正方形和长方形，他们各有好处，但常用的是正方形，尤其是在矩阵乘法这种情景下。

**正方形**

可以从arithmetic intensity的角度理解

$$
\text{arithmetic intensity} = \frac{\text{Work Done}}{\text{Data Loaded}}
$$

- 一个thread block在一定时间内完成的计算正比于面积。

- 而需要数据迁移量正比于边长之和。因为每计算一个block，在tiled matmul的场景下，都要从global memory迁移`TILE_M * TILE_K + TILE_K * TILE_N`的数据。

所以公式也可以写成

$$
\text{arithmetic intensity} = \frac{\text{TILE\_M} * \text{TILE\_N}}{\text{TILE\_M} + \text{TILE\_N}}
$$

正方形可以取到最大值。

**长方形**

在实际操作中，虽然长方形不能达到最优，但是由于warp size是32，一边长32的长方形有利于coalesced访问。

总的来说，优化可以从正方形入手。

### 数据迁移

数据迁移到SMEM时，是`TILE_M * TILE_N`的线程搬运`TILE_M * TILE_K + TILE_K * TILE_N`的数据，所以如果线程安排不当，会有线程冗余。现在的数据和block设置都被可以整除，所以无需顾虑。搬运策略如下：

- 把`tileA`和`tileB`按thread block的形状分割，每个thread负责每个对应相对位置的数据，超出边界的用if判断不做操作（现在不会）

### cuda shared memory

当使用SMEM时，有两种分配的方式，dynamic和static。如果使用static方式，则会有48kB/thread_block的上限；如果使用dynamic方式，最初只能分配一个1维的大数组再分割，这样会多出很多下标计算overhead，也不能利用编译器对2维数组的优化。


### 编译

要注意的是，在编译传统cpp和cuda cpp文件复合的可执行文件时，cmake会选择gcc作为编译器，这时无法传入nvcc特有的flag，所以需要将传统cpp和cuda cpp分开编译成lib，再link。

另外，传入nvcc特有的flag时，需要分开写，不能包括在双引号中，不然会被当做unknown flag被传给gcc

## 优化bank conflict

在简单查看`ncu --set roofline -o gemm_report -f ./build/cuda_gemm`和`ncu --set detailed -o gemm_report -f ./build/cuda_gemm`的报告后，没有显示bank conflict

## 最后

现在的分块矩阵乘法有很多限制:

```cpp
assert(tile_M == tile_N);
assert(tile_K % tile_M == 0);
assert(tile_K % tile_N == 0);
assert(N % tile_N == 0);
assert(M % tile_M == 0);
assert(K % tile_K == 0);
```

主要的问题是搬运数据时线程的分配，目前数据和形状都比较简单，暂不改写，未来的改写方法：对所有数据统一编号，按照统一编号分配给每个线程

# 利用寄存器的空间：1D thread tiling

每个thread block可以用满一个SM的65536个寄存器，总共有256kB空间。现在通过nsight compute发现每个thread只用了32个，即一个thread block用了32768个，而且现在的roofline显示是memory bound，那么我们可以考虑利用更多的register，使每个thread计算更多的数据，减轻global mem到shared mem的压力，一部分thread block搬运数据时另一部分可以计算，而不用等待下一轮搬运。

核心思想是

- 用更小的thread block来处理原来一样多的数据

- 每个thread负责大于一个的数据，重复利用register的次数更多，从而使得整体arithmetic intensity更高

- 引入thread tile，利用单个线程负责多个元素计算，增加计算访存比；当TM=8时，每执行共享内存As的8个次访存指令和共享内存Bs的1个访存指令，可执行8次计算指令，相比初始版本的计算访存比1:2，提高至8:9，有效隐藏访存延迟；

但是在此之前需要使得原来的tiled matmul支持更多形状的矩阵分块。

