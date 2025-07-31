# 学习路径

1. 了解基本架构信息和CUDA编程模型，如SM、warp、thread block、global memory、shared memory等

2. 阅读https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/，至少理解和cuda core相关的前半部分

3. 观看一部分[【CUDA进阶】Cutlass软件抽象分层与源码浅析（已完结）](https://www.bilibili.com/video/BV1kToTY6Eh5/?share_source=copy_web&vd_source=cf3292f1e44b8d7a154f3a2c429b42d3)，直到理解其中概念有困难

4. 调试运行或阅读一个cutlass example，主要搞清楚

    - （1）c++模板间参数是怎么传递的，
    
    - （2）用到了哪些类，
    
    - （3）过程中各个类的含义，
    
    - （4）最底层函数是怎么进行乘法和数据迁移的

    坚持到（4），搞不清楚，直接进行下一步再回来

5. 观看全部[【CUDA进阶】Cutlass软件抽象分层与源码浅析（已完结）](https://www.bilibili.com/video/BV1kToTY6Eh5/?share_source=copy_web&vd_source=cf3292f1e44b8d7a154f3a2c429b42d3)

# CUTLASS 简单矩阵乘法（examples/00_basic_gemm/basic_gemm.cu）

## 变量名含义

- `lda`、`ldb`等，指的是