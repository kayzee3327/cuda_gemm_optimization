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

- `lda`、`ldb`等，指的是leading dimension of ...

- k开头变量名意思是count of ...或者number of ...

## from GMEM to SMEM

### 背景信息

1. 在`cutlass::gemm::device::Gemm`中，定义了`ThreadblockShape`和`WarpShape`

    - `ThreadblockShape`: 每个thread block tile的形状大小，此处默认值是`GemmShape<128, 128, 8>`

    - `WarpShape`：每个warp tile的形状大小，此处默认值是`GemmShape<32, 64, 8>`

2. kernel launch参数

    ```cpp
    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    ```

3. `kThreads` (`kThreadCount`) 的值是`(128 / 32) * (128 / 64) * (8 / 8) * 32 == 256`

    ```cpp
    // include/cutlass/gemm/threadblock/default_mma_core_simt.h
    // Shape是ThreadblockShape即GemmShape<128, 128, 8>，
    // WarpShape是WarpShape即GemmShape<32, 64, 8>
    // PartitionsK = 1
    using WarpCount = GemmShape<
        Shape::kM / WarpShape::kM,
        Shape::kN / WarpShape::kN,
        PartitionsK
    >;

    // include/cutlass/gemm_coord.h
    static int const kMNK = M * N * K;
    static int const kCount = kMNK;

    // include/cutlass/gemm/threadblock/default_mma_core_simt.h
    // kWarpSize = 32
    // kThreads是一个Thread block中Warp Tile的数量乘32
    static int const kThreads = WarpCount::kCount * kWarpSize;
    ```

4. `PitchLinearStripminedThreadMap`位于include/cutlass/transform/pitch_linear_thread_map.h，是默认使用的thread map（从GMEM到SMEM数据搬运方式），其中

    - `kElementsPerAccess`值为1，与向量化存取相关

    - `Detail::ShapeVec`指Shape of the tile in units of vectors，是向量化存取时的thread block形状大小，这里则等于thread block tile shape

    - `Iterations`指每个thread在搬运数据时如何循环，循环几次，在编译期计算所得：

        - 首先比较`Threads >= Detail::ShapeVec::kContiguous`，以下省略`Detail::ShapeVec::`

        - 如果true，得到`layout::PitchLinearShape<1, ceil(kStrided / (kThreads / kContiguous))>`, 在定义ShapeVec时，会检查kThreads和kContiguous之间是否有整除关系，此处是`<1, 1>`

            ```cpp
            static_assert((Threads < ShapeVec::kContiguous && !(ShapeVec::kContiguous % kThreads)) ||
                            (!(kThreads % ShapeVec::kContiguous)),
                            "Shape must be divisible by number of iterations of each thread.");
            ```

        - 如果false，得到layout::PitchLinearShape`<kContiguous / kThreads, kStrided>`

5. `AccessType`是很小的一个数组，用作指针从连续内存中一次取出几个元素

### 主要的函数

```cpp
transform::threadblock::PredicatedTileIterator::load_with_byte_offset(Fragment &frag, LongIndex byte_offset)
// called from
transform::threadblock::PredicatedTileIterator::load(Fragment &frag)
// called from
gemm::threadblock::MmaPipelined::prologue(IteratorA &iterator_A, IteratorB &iterator_B, int &gemm_k_iterations)
```

1. 声明了一个`AccessType*`指针指向了`Fragment`对象

    ```cpp
    // include/cutlass/transform/threadblock/predicated_tile_iterator.h
    /// Fragment object to be loaded or stored
    using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    
    // include/cutlass/transform/threadblock/predicated_tile_iterator.h
    using AccessType = AlignedArray<Element, AccessSize, (AccessSize * sizeof_bits<Element>::value / 8)>;
    ```

    - `ThreadMap::Iterations::kCount`是1，因为`Iterations`的形状大小是thread block shape，则`kStrided / (kThreads / kContiguous)`为1

    - `ThreadMap::kElementsPerAccess`是1，不做向量化访存

    - `AccessSize`是Gemm层传入的`kAlignmentA`或`kAlignmentB`，值为1

2. 三重循环

    - s: range(ThreadMap::Iterations::kStrided)
    
    - c: range(ThreadMap::Iterations::kContiguous)

    - v: range(kAccessesPerVector)

    `idx = [s][c][v] = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous)`

3. 三重循环中，首先通过`PredicatedTileAccessIterator::set_iterator_index()`设置这个类的

    - `iteration_vector_ = index % kAccessesPerVector`值为0

    - `iteration_contiguous_`，`iteration_strided_`，指thread在ThreadMap中对应的下标

        ```cpp
        // include/cutlass/transform/threadblock/predicated_tile_access_iterator.h
        // PredicatedTileAccessIteratorPredicates::set_iterator_index()
        int residual_access = index / kAccessesPerVector; //对应到第几个thread
        iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
        iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
        ```

4. `address_iterator_.get()`是`PredicatedTileAccessIterator::get()`，此时`GatherA`和`GatherB`都设置为false，`Permute`是`layout::NoPermute`，则直接返回AccessType指针：

    ```cpp
    return reinterpret_cast<AccessType *>(
        pointer_ + 
        the_predicates.iteration_contiguous_ * (ThreadMap::Delta::kContiguous * 
        sizeof_bits<Element>::value) / 8) + the_predicates.iteration_vector_;
    ```

    - `pointer_`是矩阵的起始位置，传入值是`TensorRef::data()`

    - `ThreadMap::Delta::kContiguous`是指每行的thread相隔多远，算是一种stride，此处没有向量化访存，值为1

    那么返回的指针就是指向`pointer_`后第`the_predicates.iteration_contiguous_`*4 + `the_predicates.iteration_vector_`个byte位置

5. 此时发现，取数据的地址并没有涉及到`kStrided`，为什么呢？需要搞清楚这个load函数是在load什么。

    回到调用load处，即`gemm::threadblock::MmaPipelined::prologue()`，调用时传入了`FragmentA`，是`IteratorA::Fragment`，找到

    ```cpp
    // include/cutlass/transform/threadblock/predicated_tile_iterator.h
    using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    ```

    那么可以确定，这个Fragment的大小是由ThreadMap指定：总共取多少次$\times$每次取几个（这里是1），所以在load函数里会和`ThreadMap::Iterations`配合使用。与之前通过`AccessType*`访存相呼应，大小相同。

    - 但`FragmentC`是定义在`MmaSimt::ThreadMma`中的，意义是？

        ```cpp
        GemmShape<
            Shape::kM / Policy::WarpShape::kRow,
            Shape::kN / Policy::WarpShape::kColumn,
            Policy::LaneMmaShape::kK>,
        ```

    回到`FragmentA`和`FragmentB`的讨论，现在去看`ThreadMap::Iterations`是怎么分配数据到线程的。判断总Thread数大于Shape of the tile in units of vectors（此处是thread block shape tile）的kContiguous（以下的Contiguous方向、Strided方向指的都是thread block tile的access数相关的）

    - true：每个thread在Contiguous方向会只负责一个element，记作`Iterations::kContiguous`

        因为thread数量大于kContiguous，那么一个thread block就会处理好几段Contiguous，cutlass选择每个thread block处理`group_size = floor(kThreads / Detail::ShapeVec::kContiguous)`个Contiguous段，多余的线程不设计在ThreadMap中，再将这样的pattern重复`ceil(Detail::ShapeVec::kStrided / group_size)`次，记作`Iterations::kStrided`

    - false：每个thread在Contiguous方向负责`floor(Detail::ShapeVec::kContiguous / kThreads)`个element，记作`Iterations::kContiguous`多余的线程不设计在ThreadMap中

        每次在Strided方向上都要重复一次，`Detail::ShapeVec::kStrided`记作`Iterations::kStrided`。

    现在再看三重循环的坐标意义，就是每个thread在自己的“任务”，即ThreadMap上循环，在每次取数据后都会调用`++address_iterator_;`，而开头提到的“取数据的地址并没有涉及到`kStrided`”就是在这里实现的：

    - 这个`iterator++`会在`kAccessPerVector`、`kContiguous`、`kStrided`逐步增加，循环到每Contiguous段最后一个元素时，调用++会将pointer_加上一个数

        ```cpp
        pointer_ += params_.inc_strided_;
        ```

        使得pointer_在下一行的开始位置
    
    所以在计算地址时不需要加上iteration_strided了

6. `global_load`则是通过之前计算的predicates来知道是否超出边界，利用`@p`加指令来提高效率，这里终于抵达了ptx指令



    




    


