### 上手写一个 NCCL Demo

直接使用 NCCL API 的 C++/CUDA 代码可以让你最直观地理解其工作方式。

#### 1. 环境准备

1. **硬件**：一台拥有至少 2 个 NVIDIA GPU 的服务器。如果 GPU 间有 NVLink 连接，效果更佳。
    
2. **软件**：
    
    - 安装 NVIDIA 驱动。
        
    - 安装 CUDA Toolkit。
        
    - 安装 NCCL 库。通常，可以通过 apt、yum 或直接从 NVIDIA 官网下载安装包来安装。
        
        codeBash
        
        ```
        # 在 Ubuntu 上
        sudo apt-get install libnccl2 libnccl-dev
        ```
        

#### 2. Demo 代码：一个简单的 All-Reduce 示例 (C++/CUDA)

这个 Demo 将会在多个 GPU 上分别创建一个数据数组，然后使用 NCCL 的 ncclAllReduce 对这些数组进行求和，最后验证每个 GPU 上的结果是否正确。

**文件名：simple_all_reduce.cu**

codeC++

```
#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <unistd.h> // for gethostname
#include <vector>

// 宏，用于检查 CUDA API 调用是否成功
#define CUDACHECK(cmd) do { \
    cudaError_t e = cmd; \
    if(e != cudaSuccess) { \
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 宏，用于检查 NCCL API 调用是否成功
#define NCCLCHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("Failed: NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


int main(int argc, char* argv[]) {
    int nGpus = 0;
    CUDACHECK(cudaGetDeviceCount(&nGpus));

    if (nGpus < 2) {
        printf("This test requires at least 2 GPUs. Found %d.\n", nGpus);
        return 0;
    }

    printf("Found %d GPUs.\n", nGpus);

    // 1. 初始化每个 GPU 上的资源
    std::vector<int*> d_buffers(nGpus);       // 设备端 (GPU) 缓冲区指针
    std::vector<int*> h_buffers(nGpus);       // 主机端 (CPU) 缓冲区指针
    std::vector<cudaStream_t> streams(nGpus); // 每个 GPU 一个 CUDA 流
    const int data_size = 1024 * 1024;        // 1M integers

    for (int i = 0; i < nGpus; ++i) {
        CUDACHECK(cudaSetDevice(i));

        // 分配设备和主机内存
        CUDACHECK(cudaMalloc(&d_buffers[i], data_size * sizeof(int)));
        CUDACHECK(cudaMallocHost(&h_buffers[i], data_size * sizeof(int)));

        // 创建 CUDA 流
        CUDACHECK(cudaStreamCreate(&streams[i]));
        
        // 初始化数据，第 i 个 GPU 的数据全部为 i+1
        for (int j = 0; j < data_size; ++j) {
            h_buffers[i][j] = i + 1;
        }

        // 将初始化数据从主机拷贝到设备
        CUDACHECK(cudaMemcpyAsync(d_buffers[i], h_buffers[i], data_size * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
    }

    // 等待所有初始化拷贝完成
    for (int i = 0; i < nGpus; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    printf("Data initialization complete.\n");

    // 2. 初始化 NCCL
    ncclComm_t* comms = new ncclComm_t[nGpus];
    NCCLCHECK(ncclCommInitAll(comms, nGpus, nullptr)); // 使用所有找到的 GPU 初始化通信域
    printf("NCCL communicators initialized.\n");

    // 3. 执行 All-Reduce 操作
    // 调用 ncclAllReduce，它将所有 GPU 上的 d_buffers 的数据相加，
    // 然后将结果写回到每个 GPU 的 d_buffers 中。
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGpus; ++i) {
        NCCLCHECK(ncclAllReduce((const void*)d_buffers[i], (void*)d_buffers[i], data_size, ncclInt, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    printf("ncclAllReduce operation enqueued.\n");

    // 4. 将结果从设备拷贝回主机并验证
    for (int i = 0; i < nGpus; ++i) {
        CUDACHECK(cudaMemcpyAsync(h_buffers[i], d_buffers[i], data_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

    // 等待所有操作完成
    for (int i = 0; i < nGpus; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    printf("Result copy-back complete. Verifying...\n");

    // 计算预期的结果
    int expected_sum = 0;
    for (int i = 0; i < nGpus; ++i) {
        expected_sum += (i + 1);
    }
    printf("Expected sum for each element: %d\n", expected_sum);

    // 验证每个 GPU 上的结果
    for (int i = 0; i < nGpus; ++i) {
        bool success = true;
        for (int j = 0; j < data_size; ++j) {
            if (h_buffers[i][j] != expected_sum) {
                printf("Verification FAILED on GPU %d at element %d! Got %d, Expected %d\n", i, j, h_buffers[i][j], expected_sum);
                success = false;
                break;
            }
        }
        if (success) {
            printf("Verification PASSED on GPU %d.\n", i);
        }
    }

    // 5. 清理资源
    for (int i = 0; i < nGpus; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_buffers[i]));
        CUDACHECK(cudaFreeHost(h_buffers[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
    delete[] comms;

    return 0;
}
```

#### 3. 编译和运行

1. **编译**：使用 nvcc 编译器，并链接 nccl 和 cudart 库。
    
    codeBash
    
    ```
    nvcc simple_all_reduce.cu -o simple_all_reduce -lnccl
    ```
    
2. **运行**：直接执行编译好的程序。
    
    codeBash
    
    ```
    ./simple_all_reduce
    ```
    

**预期输出 (以 4-GPU 系统为例):**

codeCode

```
Found 4 GPUs.
Data initialization complete.
NCCL communicators initialized.
ncclAllReduce operation enqueued.
Result copy-back complete. Verifying...
Expected sum for each element: 10
Verification PASSED on GPU 0.
Verification PASSED on GPU 1.
Verification PASSED on GPU 2.
Verification PASSED on GPU 3.
```

在这个例子中，GPU 0 的数据是 1，GPU 1 是 2，GPU 2 是 3，GPU 3 是 4。All-Reduce 求和后，每个元素都应该是 1+2+3+4 = 10。程序最后验证了所有 GPU 上的结果都正确。

---

### 学习总结与下一步

通过这个 Demo，你亲手调用了 NCCL 的核心 API，并理解了其基本工作流程：**初始化通信域 -> 在 CUDA 流上入队 NCCL 操作 -> 同步流 -> 验证结果**。

**下一步可以探索的：**

1. **尝试其他集合操作**：将 ncclAllReduce 换成 ncclBroadcast, ncclAllGather 等，观察它们的不同行为。
    
2. **多进程/多节点**：这个 Demo 是单进程多 GPU 的。真正的分布式训练是多进程多节点的。可以研究如何使用 MPI 结合 NCCL 来编写跨节点的程序。ncclCommInitRank 是用于多进程场景的初始化函数。
    
3. **性能分析**：使用 nsys 或 ncu 等 NVIDIA 的性能分析工具来剖析你的 Demo，观察 NCCL 操作的耗时，以及它是否与计算 Kernel 发生了重叠。
    
4. **阅读框架源码**：深入 PyTorch 的 torch.distributed 部分，看看它是如何封装和调用 NCCL 的。这是从“使用”到“精通”的必经之路。