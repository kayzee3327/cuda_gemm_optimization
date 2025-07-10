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

# 实现简单的cuda矩阵乘

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








    




