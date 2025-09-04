
### Runtime API - 与 GPU 对话的语言

CUDA Runtime API 是一套 C/C++ 函数库，它为开发者提供了一个**高层次的、抽象的接口**，用于管理和控制 GPU。它是你编写的大部分 CUDA 程序的“主心骨”，也是 cuBLAS 和 cuDNN 等高层库构建自身功能的基石。

可以把它想象成一个**“GPU 操作系统”的系统调用接口 (System Call Interface)**。你通过它，向 NVIDIA 驱动和 GPU 硬件下达指令。

#### **第一幕：Runtime vs. Driver API - 两种对话方式**

在 CUDA 的世界里，其实存在两套 API 可以与 GPU 对话：

1.  **Runtime API**:
    *   **头文件**: `cuda_runtime.h`
    *   **库**: `libcudart.so`
    *   **特点**:
        *   **高层次、易用**: 封装了大量复杂的细节。例如，它会自动处理 CUDA 上下文（Context）的创建和管理。
        *   **隐式初始化**: 你调用的第一个 CUDA Runtime 函数（如 `cudaMalloc`）会自动初始化驱动、设置设备、创建上下文。
        *   **面向开发者**: 是绝大多数 CUDA 应用程序开发的首选。

2.  **Driver API**:
    *   **头文件**: `cuda.h`
    *   **库**: `libcuda.so` (它实际上是 NVIDIA 驱动的一部分)
    *   **特点**:
        *   **低层次、复杂、强大**: 提供了对 GPU 更精细的控制。你需要**手动管理**上下文（`CUcontext`）、模块（`CUmodule`）、函数（`CUfunction`）的加载和执行。
        *   **显式初始化**: 必须先调用 `cuInit(0)`。
        *   **面向框架和驱动开发者**: PyTorch、TensorFlow 等框架的底层，或者需要动态加载和编译 CUDA 代码的场景，通常会使用 Driver API。

**关系**: CUDA Runtime API 在底层实际上是**对 Driver API 的一层封装**。你调用 `cudaMalloc`，它在内部可能会调用一系列 `cu...` 开头的 Driver API 函数来完成工作。

**我们的学习重点将是 Runtime API**，因为它更常用，也更能代表通用的 CUDA 编程模式。

#### **第二幕：Runtime API 的核心功能 - 四大支柱**

CUDA Runtime API 的功能可以被归纳为四大支柱，它们构成了你与 GPU 交互的全部流程。

**1. 设备管理 (Device Management)**
这是你与 GPU 建立连接的第一步。

*   `cudaGetDeviceCount(int* count)`: 查询系统中有多少个可用的 NVIDIA GPU。
*   `cudaSetDevice(int device)`: **关键！** 将后续所有的 CUDA 操作（内存分配、kernel 启动等）**绑定到指定的 GPU 设备**上。在多 GPU 编程中，这是切换操作目标的核心函数。
*   `cudaGetDevice(int* device)`: 获取当前正在使用的设备 ID。
*   `cudaGetDeviceProperties(cudaDeviceProp* prop, int device)`: 获取指定 GPU 的详细属性，如设备名称、全局内存大小、SM 数量、CUDA 版本兼容性等。这对于编写可移植的、能自适应硬件的代码至关重要。

**2. 内存管理 (Memory Management)**
这是我们最熟悉的部分，但其背后有更深的含义。

*   `cudaMalloc(void** devPtr, size_t size)`: 在**当前设备**的**全局内存 (Global Memory)** 中分配 `size` 字节的线性内存。
*   `cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)`: 在不同类型的内存之间拷贝数据。`kind` 参数是核心：
    *   `cudaMemcpyHostToDevice`: CPU -> GPU
    *   `cudaMemcpyDeviceToHost`: GPU -> CPU
    *   `cudaMemcpyDeviceToDevice`: 同一个或不同 GPU 之间
    *   `cudaMemcpyHostToHost`: CPU 内部（虽然很少这样用）
*   `cudaFree(void* devPtr)`: 释放 GPU 内存。
*   **更高级的内存**:
    *   `cudaMallocManaged()`: 分配**统一内存 (Unified Memory)**。这种内存对 CPU 和 GPU 都是可见的，系统会自动处理数据在两者之间的迁移。它简化了编程，但性能可能低于手动管理。
    *   `cudaMallocHost()` / `cudaHostAlloc()`: 分配**页锁定内存 (Page-locked or Pinned Memory)**。这种 CPU 内存被“锁定”，不能被操作系统交换到磁盘。**使用页锁定内存作为 `cudaMemcpy` 的源或目标，可以获得更高的传输带宽**，因为它允许 CUDA 驱动使用 DMA (Direct Memory Access) 来进行异步传输。这是性能优化的一个关键技巧。

**3. 执行控制 (Execution Control)**
这是你命令 GPU “开始工作”的地方。

*   `kernel_name<<<gridDim, blockDim, sharedMemSize, stream>>>(...)`: **Kernel 启动语法**。这是 CUDA C++ 的扩展，不是一个函数调用。它指示 Runtime 将一个名为 `kernel_name` 的 `__global__` 函数（我们称之为 Kernel）在 GPU 上执行。
    *   `gridDim`: 定义了要启动的 **线程块 (Block)** 的网格维度。
    *   `blockDim`: 定义了每个线程块中包含的 **线程 (Thread)** 的维度。
    *   `sharedMemSize` (可选): 为每个块动态分配的共享内存大小。
    *   `stream` (可选): 指定这个 Kernel 在哪个 CUDA Stream 上执行。
*   `cudaDeviceSynchronize()`: **关键！** 这是一个**阻塞主机 (CPU) 的函数**。它会使 CPU 线程暂停，直到**该设备上之前所有被启动的 CUDA 操作**（包括 kernel 执行和内存拷贝）全部完成。这在需要确保 GPU 结果可用时（如拷贝回 CPU 前）或进行精确计时时，是必不可少的。

**4. Stream 与事件管理 (Stream and Event Management)**
这是 CUDA 并发编程和性能优化的核心。

*   **CUDA Stream (`cudaStream_t`)**:
    *   **概念**: 一个 Stream 就是 GPU 上的一个**任务队列**。所有被提交到同一个 Stream 的操作，都将**按顺序执行**。
    *   **并发**: **不同 Stream 上的操作，可以并发执行**（只要硬件资源允许）。
    *   **默认 Stream**: 如果你不指定 Stream，所有操作都会被提交到默认的 `stream 0`。默认 Stream 有一个特殊的同步行为：它会等待所有其他 Stream 的任务完成后才开始，并且所有其他 Stream 也会等待它完成。
    *   **创建**: `cudaStreamCreate(&stream)`。
    *   **销毁**: `cudaStreamDestroy(stream)`。

*   **CUDA Event (`cudaEvent_t`)**:
    *   **概念**: 一个 Event 就像是 Stream 中的一个“标记点”或“里程碑”。它可以在某个时间点被“记录”到 Stream 中。
    *   **用途**:
        1.  **精确计时**: 在操作前后分别记录两个 Event，然后使用 `cudaEventElapsedTime()` 计算它们之间的时间差，可以精确测量 GPU 操作的耗时，不受 CPU 阻塞的影响。
        2.  **Stream 间同步**: 一个 Stream 可以被设置为 `cudaStreamWaitEvent()`，它会暂停执行，直到另一个 Stream 中被记录的某个 Event 完成。这是实现复杂的、有依赖关系的任务流水线的关键。
    *   **创建/记录/同步**: `cudaEventCreate()`, `cudaEventRecord()`, `cudaEventSynchronize()`, `cudaStreamWaitEvent()`.

#### **第三幕：一个完整的 Runtime API 工作流**

让我们用一个简单的向量加法例子，串联起所有这些概念。

```c++
#include <iostream>

// Kernel: 在 GPU 上执行的函数
__global__ void vecAdd(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // --- 1. 设备管理 ---
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) { // 如果没有可用的GPU设备，打印错误信息并退出程序。
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    cudaSetDevice(0); // 选择使用第一个 GPU (device 0)

    // --- 2. 内存管理 ---
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C; // Host pointers
    float *d_A, *d_B, *d_C; // Device pointers

    // 分配 CPU 内存
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    // 初始化 CPU 数据
    for(int i=0; i<N; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // 分配 GPU 内存
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将数据从 CPU 拷贝到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // --- 3. 执行控制 ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 Kernel！
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // --- 等待并获取结果 ---
    cudaDeviceSynchronize(); // 阻塞 CPU，直到 vecAdd kernel 完成

    // 将结果从 GPU 拷贝回 CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证...
    if (h_C[0] != 3.0f) std::cout << "Verification FAILED!" << std::endl;
    else std::cout << "Verification PASSED!" << std::endl;

    // --- 清理 ---
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

---

**今日总结与回顾**:

今天，我们下潜到了 CUDA 编程的核心——**Runtime API**。

*   我们理解了它是与 GPU 交互的**高级抽象层**，并将其与更底层的 Driver API 进行了区分。
*   我们系统地学习了它的四大支柱功能：**设备管理、内存管理、执行控制、以及 Stream/Event 并发**。
*   我们认识到，像 `cudaSetDevice`, `cudaDeviceSynchronize`, **页锁定内存**, 以及 **CUDA Stream** 这些概念，是编写高性能、多 GPU 程序的基础。
*   通过一个完整的实例，我们将这些 API 调用串联起来，构建了一个从数据准备到执行再到验证的完整 CUDA 程序。

你现在理解了 cuBLAS 和 cuDNN 这些高层库能够工作的底层基础。你不再仅仅知道“数据需要在 GPU 上”，而是知道了如何**查询设备、选择设备、分配和传输数据、以及如何启动计算和同步结果**。

在我们的下一部分，我们将聚焦于 Runtime API 中最高级、也最能体现性能优化艺术的部分——**使用 CUDA Stream 和 Event 来实现计算与数据传输的重叠 (Overlap)**。这将是我们榨干硬件最后一点性能的关键。