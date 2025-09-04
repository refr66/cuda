### **学习路线图总览**

*   **第一阶段：CUDA 基础入门 (The Foundation)**
    *   **目标：** 理解CUDA编程模型，能够独立编写、编译和运行简单的CUDA程序，掌握基本的数据传输和核函数设计。
    *   **成果：** 能够实现并行的基础算法，如向量加法、矩阵运算。

*   **第二阶段：CUDA 进阶与性能优化 (The Optimizer)**
    *   **目标：** 掌握GPU内存层次结构，学会使用性能分析工具定位瓶颈，并运用共享内存、流等高级特性进行深度优化。
    *   **成果：** 能够将一个朴素的CUDA程序性能提升数倍甚至数十倍，理解性能背后的硬件原理。

*   **第三阶段：CUDA 大师之路 - AI 与系统级应用 (The Master)**
    *   **目标：** 掌握现代GPU的尖端特性（如Tensor Cores），能够手写高性能AI算子，并将其集成到主流深度学习框架中，具备系统级的优化视野。
    *   **成果：** 能够从零开始为AI模型开发自定义的高性能CUDA算子，并利用TensorRT等工具进行端到端的推理优化。

---

### **第一阶段：CUDA 基础入门 (The Foundation)**

**核心模块：** [[CUDA编程模型]], [[GPU 架构 (初级)]], [[编译与API]]

| 核心内容 | 学习要点 | 实践 Demo |
| :--- | :--- | :--- |
| **1. 环境搭建与Hello World** | - 安装CUDA Toolkit, NVIDIA Driver<br>- 配置IDE (VS Code / Visual Studio)<br>- 理解`nvcc`编译流程 | **Demo 1.0: CUDA版 "Hello World"**<br>编写一个最简单的Kernel，在GPU上打印出线程ID，验证环境配置成功。 |
| **2. CUDA编程模型核心** | - **线程层次结构**: Grid, Block, Thread<br>- **核函数 (Kernel)**: `__global__` 函数的定义与调用<br>- **线程索引**: 如何使用 `threadIdx`, `blockIdx`, `blockDim` 计算全局唯一ID | **Demo 1.1: 向量加法 (Vector Addition)**<br>最经典的入门程序。实现 `C = A + B`，其中A, B, C都是大型向量。学习：<br>1. 分配GPU内存 (`cudaMalloc`)<br>2. H2D数据拷贝 (`cudaMemcpyHostToDevice`)<br>3. 启动Kernel<br>4. D2H数据拷贝 (`cudaMemcpyDeviceToHost`)<br>5. 释放内存 (`cudaFree`) |
| **3. GPU架构与内存模型(初级)** | - **基本架构**: SM (Streaming Multiprocessor), SP (Streaming Processor/CUDA Core)<br>- **基本内存模型**: Host Memory vs. Device Memory (Global Memory) | **Demo 1.2: 矩阵加法 (Matrix Addition)**<br>将向量加法扩展到二维。学习：<br>1. 如何将2D问题映射到1D的Grid/Block上<br>2. 二维数据的索引计算方法 |

**第一阶段成果:**
*   熟练掌握 `cudaMalloc`/`cudaMemcpy`/`cudaFree` 等核心API。
*   能够独立编写和启动简单的核函数，并正确处理线程索引。
*   对GPU并行计算的模式有初步的、直观的理解。

---

### **第二阶段：CUDA 进阶与性能优化 (The Optimizer)**

**核心模块：** [[性能分析与调试]], [[CUDA 核心工作实践]], [[GPU 架构 (深入)]]

| 核心内容 | 学习要点 | 实践 Demo |
| :--- | :--- | :--- |
| **1. 性能分析工具** | - **NVIDIA Nsight Systems**: 用于分析应用程序的系统级性能，查看API调用和数据传输。<br>- **NVIDIA Nsight Compute**: 用于对单个Kernel进行深度分析，查看内存、计算等瓶颈。 | **Demo 2.1: 朴素的矩阵乘法 (Naive Matrix Multiplication)**<br>编写一个最直观、但性能极差的矩阵乘法Kernel。每个线程计算结果矩阵的一个元素。以此作为后续优化的基准(Baseline)。 |
| **2. 优化关键：内存** | - **GPU内存层次**: Global, L1/L2 Cache, **Shared Memory**, Constant Memory, Registers<br>- **全局内存访问**: Coalesced Access (合并访问)的重要性<br>- **共享内存**: `__shared__` 关键字，块内线程高速通信，减少全局内存访问 | **Demo 2.2: 使用共享内存优化的矩阵乘法 (Tiled Matrix Multiplication)**<br>这是CUDA优化的“圣杯”级示例。通过将输入矩阵分块(Tile)加载到共享内存中，大幅减少对全局内存的访问次数。这是**最重要的一个Demo**。 |
| **3. 性能分析实践** | - 使用Nsight Compute分析Demo 2.1和2.2。<br>- 对比二者的**内存吞吐量**、**指令延迟**、**占用率 (Occupancy)**等指标。<br>- 直观感受性能差异的原因（例如，L1/Shared Memory Hit Rate的巨大提升）。 | **Demo 2.3: 分析与验证**<br>为Demo 2.1和2.2生成性能分析报告，学习阅读报告，找出瓶颈，并验证优化（Demo 2.2）确实解决了这些瓶颈。 |
| **4. 并发与同步** | - **线程块内同步**: `__syncthreads()`<br>- **流 (Streams)**: `cudaStream_t`，实现Kernel执行与数据传输的重叠 (Overlap)。<br>- **异步API**: `cudaMemcpyAsync` | **Demo 2.4: 并行数据处理流**<br>模拟一个场景：一个大文件需要分成多个Chunk处理。使用多个CUDA Stream，让GPU在计算Chunk N时，CPU同时在准备并传输Chunk N+1的数据，实现计算和数据传输的并行。 |
| **5. 计算优化** | - **Warp与线程发散 (Divergence)**: 理解`if-else`对性能的影响。<br>- **原子操作 (Atomic Operations)**: `atomicAdd()`等，用于并行归约(Reduction)等场景。 | **Demo 2.5: 并行归约 (Parallel Reduction)**<br>实现一个高效的求和算法。从一个朴素的、有数据竞争的版本开始，逐步优化：<br>1. 使用原子操作解决冲突。<br>2. 使用共享内存进行块内归约，最后再用原子操作写回全局内存，大幅提升性能。 |

**第二阶段成果:**
*   能够熟练使用Nsight工具定位性能瓶颈。
*   精通共享内存的使用，并理解其对性能的决定性作用。
*   掌握CUDA Streams，实现计算与通信的重叠。
*   理解Warp级执行、线程发散等更深层次的硬件行为。

---

### **第三阶段：CUDA 大师之路 - AI 与系统级应用 (The Master)**

**核心模块：** [[现代 GPU 的硬件特性]], [[算子级深度优化]], [[系统级与框架级视角]], [[推理优化]], [[库与模板]]

| 核心内容 | 学习要点 | 实践 Demo |
| :--- | :--- | :--- |
| **1. 现代GPU硬件加速** | - **Tensor Cores**: 专门用于`D = A * B + C`混合精度矩阵运算的硬件单元，是AI加速的核心。<br>- **WMMA API / CUTLASS**: 学习使用NVIDIA官方提供的API或模板库来驾驭Tensor Cores。 | **Demo 3.1: 基于Tensor Core的矩阵乘法**<br>使用CUTLASS库（或直接用WMMA intrinsics），重写矩阵乘法。与Demo 2.2对比，感受Tensor Core带来的数倍性能飞跃。这是进入AI领域的敲门砖。 |
| **2. AI核心算子开发** | - **卷积 (Convolution)**: 理解Im2col等经典卷积实现方法，并用CUDA实现。<br>- **注意力机制 (Attention)**: 分析Self-Attention中的主要瓶颈（通常是Softmax和BMM）。 | **Demo 3.2: 2D卷积前向传播 (Forward Pass)**<br>从零开始，实现一个基础的2D卷积核函数。先实现一个朴素版本，然后应用第二阶段学到的共享内存等技巧进行优化。这将让你深刻理解cuDNN这类库的价值。 |
| **3. 框架集成** | - **PyTorch C++ Extensions**: 学习如何将你的CUDA代码编译成一个Python可调用的动态链接库。<br>- **TensorFlow Custom Ops**: 类似的技术，为TF编写自定义算子。 | **Demo 3.3: 为PyTorch编写自定义激活函数**<br>编写一个自定义的激活函数（如Mish或其变体）的CUDA Kernel，并将其封装成PyTorch的扩展。你将可以在Python中像调用`torch.relu`一样调用你的`my_lib.mish`，并享受GPU加速。 |
| **4. 库与系统级优化** | - **cuBLAS, cuDNN, cuSPARSE**: 了解何时应该“造轮子”，何时应该使用NVIDIA官方的高性能库。<br>- **Kernel Fusion**: 将多个连续的、访存密集的小Kernel合并成一个大Kernel，减少Kernel启动开销和全局内存读写。 | **Demo 3.4: 手动实现Kernel Fusion**<br>实现一个 `y = a*x + b` (AXPY) 和一个 `z = relu(y)` 的操作。先用两个独立的Kernel实现，然后将它们融合成一个`z = relu(a*x + b)`的Kernel。使用Nsight分析，对比性能提升。 |
| **5. 推理部署优化** | - **TensorRT**: 学习NVIDIA的推理优化器和运行时。它能自动进行Kernel Fusion、精度量化、选择最优Kernel等。<br>- **模型格式**: ONNX。 | **Demo 3.5: 使用TensorRT优化一个简单模型**<br>1. 导出一个简单的ONNX模型（例如，来自PyTorch的一个小型MLP或CNN）。<br>2. 使用TensorRT C++/Python API加载模型，构建优化引擎。<br>3. 运行推理，并与原始框架（PyTorch）中的推理延迟进行基准测试，感受数量级的性能提升。 |

**第三阶段成果:**
*   具备开发和优化主流AI模型核心算子的能力。
*   能够将自定义CUDA代码无缝集成到PyTorch/TensorFlow等框架中。
*   拥有系统级的优化视野，懂得如何使用TensorRT等工具链进行端到端的模型部署优化。
*   真正成为连接算法与硬件的桥梁，成为一名稀缺的AI系统与高性能计算专家。

---

