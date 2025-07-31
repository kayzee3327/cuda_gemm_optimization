# 架构信息

1. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications

2. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x

3. https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

4. https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/

# Programming Model

1. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model

# 优化学习帖

## 矩阵乘法

1. https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

2. https://siboehm.com/articles/22/CUDA-MMM

3. https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE

## 分块矩阵乘和shared memory

1. https://penny-xu.github.io/blog/tiled-matrix-multiplication

2. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html?highlight=shared%2520memory#shared-memory

3. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x

4. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x

5. [拯救你的CUDA！什么是bank，为什么会发生bank conflict？？](https://www.bilibili.com/video/BV1Xz4y157p7/?share_source=copy_web&vd_source=cf3292f1e44b8d7a154f3a2c429b42d3)

6. https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity/

## CUTLASS

1. [必读 | 矩阵乘法也可从此入门](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

2. cutlass源码导读（1）——API与设计理念 - 霸王手枪腿的文章 - 知乎 https://zhuanlan.zhihu.com/p/588953452

3. cutlass源码导读（2）——Gemm的计算流程 - 霸王手枪腿的文章 - 知乎 https://zhuanlan.zhihu.com/p/592689326

4. cutlass源码导读（3）——核心软件抽象 - 霸王手枪腿的文章 - 知乎 https://zhuanlan.zhihu.com/p/595533802

5. cutlass源码导读（4）——软件分层与源码目录 - 霸王手枪腿的文章 - 知乎 https://zhuanlan.zhihu.com/p/600091558

6. https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html

7. [【CUDA进阶】Cutlass软件抽象分层与源码浅析（已完结）| 解释清楚了第1个博客](https://www.bilibili.com/video/BV1kToTY6Eh5/?share_source=copy_web&vd_source=cf3292f1e44b8d7a154f3a2c429b42d3)