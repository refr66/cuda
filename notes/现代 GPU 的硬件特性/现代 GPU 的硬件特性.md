- **GPU 硬件架构**
    
    - **SM (Streaming Multiprocessor)**
        
    - **CUDA Core**

现代 GPU（如 Ampere, Hopper, Ada Lovelace 架构）不仅仅是 CUDA Core 数量的增加，更重要的是引入了专门为 AI 设计的硬件单元。

*   **Tensor Cores**:
    *   **是什么**：专门用于执行矩阵乘加运算 (`D = A * B + C`) 的硬件单元，这是深度学习计算的核心。
    *   **为什么重要**：它们为 `FP16`、`BF16`、`TF32`、`INT8` 等低精度数据类型提供惊人的吞吐量，比使用标准 CUDA Core 计算 `FP32` 要快数倍甚至一个数量级。
    *   **如何使用**：
        *   **cuBLAS/cuDNN**: 这些库会自动使用 Tensor Cores（当输入的数据类型和维度满足条件时）。
        *   **CUTLASS**: 提供了精细的模板，允许你在 Warp 级别直接调用 Tensor Core 指令 (`wmma` 或 `mma` 指令)，用于构建自定义的高性能算子。
        *   **实践中**：AI 系统的开发人员必须将模型训练（混合精度）和推理（INT8 量化）与 Tensor Cores 的使用紧密结合，这是性能提升最主要的来源。

*   **结构化稀疏 (Structured Sparsity)**：
    *   **是什么**：从 Ampere 架构开始，硬件支持一种特殊的 **2:4 稀疏模式**，即在一个 4 元素的向量中，可以有两个非零值。
    *   **为什么重要**：如果你的权重矩阵能被剪枝成这种 2:4 稀疏格式，Tensor Core 可以在不损失精度的前提下，达到**2倍**的计算吞吐量。
    *   **实践中**：这需要算法（如何剪枝模型到 2:4 稀疏）和工程（使用支持稀疏的库，如 `cuSPARSELt`）的结合。

*   **异步执行与并发 (Asynchronous Execution & Concurrency)**：
    *   **CUDA Streams**: AI 系统通常是一个复杂的计算图。使用 CUDA Streams 可以让不相关的任务（如数据拷贝、不同层的计算）在 GPU 上**并行执行**，最大化硬件利用率。
    *   **Copy/Compute Overlap**: 一个经典的模式是，当 GPU 正在计算当前 batch (on Stream 1) 时，CPU 和 DMA 引擎可以异步地将下一个 batch 的数据从 Host 拷贝到 Device (on Stream 2)。这能有效隐藏数据传输的延迟。
