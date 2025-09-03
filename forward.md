好的，用 CUDA C++ 手写神经网络的前向和反向传播是一个绝佳的练习，它能让你深刻理解 GPU 是如何通过大规模并行计算来加速深度学习的。

这个任务比之前的纯 C++ 或 Python 实现要复杂得多，因为它涉及到：
1.  **大规模并行化**：将矩阵运算分解给数千个 CUDA 线程。
2.  **内存管理**：在 CPU (Host) 和 GPU (Device) 之间手动管理内存。
3.  **Kernel 设计**：为每个计算步骤（矩阵乘法、激活函数、梯度计算等）编写专门的 GPU 内核函数。

我们将沿用之前的 XOR 示例，并构建一个完整的、可运行的 `.cu` 文件。

---

### 1. 核心并行化思想

*   **矩阵乘法 (`C = A * B`)**: 这是最核心的操作。我们将启动一个 2D 的线程网格（Grid），让**每个线程负责计算输出矩阵 C 中的一个元素 `C[i, j]`**。要计算 `C[i, j]`，该线程需要对 A 的第 `i` 行和 B 的第 `j` 列做点积。
*   **元素级操作 (Element-wise Operations)**: 如激活函数 `A = sigmoid(Z)` 或向量加法 `Z = Z + b`。这类操作是**完美并行**的。我们可以启动一个线程网格，让**每个线程负责处理一个元素**。
*   **反向传播**: 反向传播中的核心计算也是矩阵乘法（例如 `dW = A_prev.T @ dZ_curr`）和元素级操作，因此可以采用与前向传播相同的并行策略。

---

### 2. Naive CUDA C++ 实现

这是一个**基础但完整**的实现。为了教学目的，它并没有使用高级优化（如共享内存、cuBLAS库），而是清晰地展示了每个线程的工作。

将以下代码保存为 `nn_cuda.cu`。

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

// CUDA 错误检查宏
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ================= KERNELS =================

__device__ float sigmoid_gpu(float z) {
    return 1.0f / (1.0f + expf(-z));
}

__device__ float sigmoid_derivative_gpu(float z) {
    float s = sigmoid_gpu(z);
    return s * (1.0f - s);
}

// Kernel for Forward Pass: Z = X * W + b, then A = sigmoid(Z)
__global__ void forward_kernel(const float* X, const float* W, const float* b, float* Z, float* A,
                               int m, int n, int p) {
    // m: rows of X (batch size)
    // n: cols of X / rows of W
    // p: cols of W (output features)
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += X[row * n + k] * W[k * p + col];
        }
        sum += b[col];
        Z[row * p + col] = sum;
        A[row * p + col] = sigmoid_gpu(sum);
    }
}

// Kernel for calculating output layer error: dZ2 = A2 - y
__global__ void output_error_kernel(const float* A2, const float* y, float* dZ2, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        dZ2[idx] = A2[idx] - y[idx];
    }
}

// Kernel for backpropagating error to hidden layer
// dZ1 = (dZ2 * W2^T) .* sigmoid_derivative(Z1)
__global__ void hidden_error_kernel(const float* dZ2, const float* W2, const float* Z1, float* dZ1,
                                    int m, int hidden_size, int output_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < hidden_size) {
        float sum = 0.0f;
        for (int k = 0; k < output_size; ++k) {
            // W2^T access pattern: W2[col * output_size + k]
            sum += dZ2[row * output_size + k] * W2[col * output_size + k];
        }
        dZ1[row * hidden_size + col] = sum * sigmoid_derivative_gpu(Z1[row * hidden_size + col]);
    }
}


// Kernel for updating weights: W = W - lr * dW
// dW is calculated as X^T * dZ
__global__ void update_weights_kernel(const float* Prev_A, const float* dZ, float* W, float* b,
                                      float learning_rate, int m, int prev_size, int curr_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to prev_size
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to curr_size

    if (row < prev_size && col < curr_size) {
        float dW = 0.0f;
        for (int k = 0; k < m; ++k) {
            // Prev_A^T access: Prev_A[k * prev_size + row]
            dW += Prev_A[k * prev_size + row] * dZ[k * curr_size + col];
        }
        
        W[row * curr_size + col] -= learning_rate * (dW / m);
    }

    // Update bias (can be done more efficiently with a separate reduction kernel)
    // Here we let threads for the first row handle it for simplicity
    if (row == 0 && col < curr_size) {
        float db = 0.0f;
        for (int k = 0; k < m; ++k) {
            db += dZ[k * curr_size + col];
        }
        b[col] -= learning_rate * (db / m);
    }
}

// ================= HOST CODE =================

void print_matrix(const std::vector<float>& mat, int rows, int cols, const std::string& name) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f ", mat[i * cols + j]);
        }
        std::cout << "\n";
    }
}

int main() {
    // --- 1. Define Network Architecture & Hyperparameters ---
    const int input_size = 2;
    const int hidden_size = 2;
    const int output_size = 1;
    const int batch_size = 4; // Number of training samples
    const int epochs = 20000;
    const float learning_rate = 0.5f;

    // --- 2. Prepare Host (CPU) Data ---
    std::vector<float> h_X = {0,0, 0,1, 1,0, 1,1};
    std::vector<float> h_y = {0, 1, 1, 0};

    // Initialize weights and biases
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-0.1, 0.1);
    std::vector<float> h_W1(input_size * hidden_size), h_b1(hidden_size);
    std::vector<float> h_W2(hidden_size * output_size), h_b2(output_size);
    for(auto& v : h_W1) v = dist(gen);
    for(auto& v : h_b1) v = 0;
    for(auto& v : h_W2) v = dist(gen);
    for(auto& v : h_b2) v = 0;

    // --- 3. Allocate Device (GPU) Memory ---
    float *d_X, *d_y, *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_Z1, *d_A1, *d_Z2, *d_A2;
    float *d_dZ1, *d_dZ2;

    gpuErrchk(cudaMalloc(&d_X, h_X.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_y, h_y.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_W1, h_W1.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b1, h_b1.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_W2, h_W2.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b2, h_b2.size() * sizeof(float)));

    // Intermediate activations and weighted sums
    gpuErrchk(cudaMalloc(&d_Z1, batch_size * hidden_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_A1, batch_size * hidden_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_Z2, batch_size * output_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_A2, batch_size * output_size * sizeof(float)));
    
    // Gradients
    gpuErrchk(cudaMalloc(&d_dZ1, batch_size * hidden_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_dZ2, batch_size * output_size * sizeof(float)));

    // --- 4. Copy Initial Data from Host to Device ---
    gpuErrchk(cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- 5. Training Loop ---
    dim3 threadsPerBlock(16, 16);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < epochs; ++i) {
        // --- FORWARD PASS ---
        // Layer 1 (Input -> Hidden)
        dim3 grid1((hidden_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        forward_kernel<<<grid1, threadsPerBlock>>>(d_X, d_W1, d_b1, d_Z1, d_A1, batch_size, input_size, hidden_size);

        // Layer 2 (Hidden -> Output)
        dim3 grid2((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        forward_kernel<<<grid2, threadsPerBlock>>>(d_A1, d_W2, d_b2, d_Z2, d_A2, batch_size, hidden_size, output_size);

        // --- BACKWARD PASS ---
        // Calculate output layer error (dZ2)
        output_error_kernel<<<(batch_size * output_size + 255) / 256, 256>>>(d_A2, d_y, d_dZ2, batch_size * output_size);

        // Calculate hidden layer error (dZ1)
        hidden_error_kernel<<<grid1, threadsPerBlock>>>(d_dZ2, d_W2, d_Z1, d_dZ1, batch_size, hidden_size, output_size);
        
        // --- UPDATE WEIGHTS ---
        // Update W2 and b2
        update_weights_kernel<<<dim3((output_size + 15)/16, (hidden_size+15)/16), dim3(16,16)>>>(
                                d_A1, d_dZ2, d_W2, d_b2, learning_rate, batch_size, hidden_size, output_size);

        // Update W1 and b1
        update_weights_kernel<<<dim3((hidden_size + 15)/16, (input_size+15)/16), dim3(16,16)>>>(
                                d_X, d_dZ1, d_W1, d_b1, learning_rate, batch_size, input_size, hidden_size);

        // Optional: Print loss periodically
        if (i % 2000 == 0) {
            std::vector<float> h_A2(batch_size * output_size);
            gpuErrchk(cudaMemcpy(h_A2.data(), d_A2, h_A2.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float loss = 0.0f;
            for(size_t j = 0; j < h_y.size(); ++j) {
                loss += 0.5f * pow(h_y[j] - h_A2[j], 2);
            }
            std::cout << "Epoch " << i << ", Loss: " << loss / h_y.size() << std::endl;
        }
    }

    gpuErrchk(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Training finished in " << diff.count() << " seconds.\n";

    // --- 6. Show Final Predictions ---
    std::vector<float> h_A2_final(batch_size * output_size);
    gpuErrchk(cudaMemcpy(h_A2_final.data(), d_A2, h_A2_final.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n--- Final Predictions ---\n";
    for(int i = 0; i < batch_size; ++i) {
        printf("Input: (%.0f, %.0f), Target: %.0f, Predicted: %.4f\n", 
            h_X[i*2], h_X[i*2+1], h_y[i], h_A2_final[i]);
    }
    
    // --- 7. Free Device Memory ---
    cudaFree(d_X); cudaFree(d_y);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_Z1); cudaFree(d_A1); cudaFree(d_Z2); cudaFree(d_A2);
    cudaFree(d_dZ1); cudaFree(d_dZ2);
    
    return 0;
}
```

---

### 3. 代码详解

*   **`__device__` 函数**: `sigmoid_gpu` 和 `sigmoid_derivative_gpu` 只能被 GPU 内核调用，不能被 CPU 调用。
*   **`forward_kernel`**:
    *   这个内核函数执行一个完整的前向传播层：`Z = XW + b` 和 `A = sigmoid(Z)`。
    *   `row` 和 `col` 唯一地标识了输出矩阵 `A` 中的一个元素，每个线程负责计算这一个元素。
    *   内部的 `for` 循环是执行点积操作。
*   **`output_error_kernel`**: 一个简单的元素级内核，计算 `dZ2 = A2 - y`。
*   **`hidden_error_kernel`**:
    *   这是反向传播的核心之一，计算 `dZ1`。
    *   它首先执行矩阵乘法 `dZ2 * W2^T`，然后将结果与 `sigmoid'(Z1)` 进行元素级相乘。
    *   注意访问 `W2` 的方式 `W2[col * output_size + k]` 是为了模拟对 `W2` 的转置（`W2^T`）的访问。
*   **`update_weights_kernel`**:
    *   这个内核计算梯度 `dW = Prev_A^T * dZ` 并直接更新 `W` 和 `b`。
    *   计算 `dW` 时，访问 `Prev_A` 的方式 `Prev_A[k * prev_size + row]` 是为了模拟 `Prev_A` 的转置。
    *   更新偏置 `b` 的部分比较粗糙，它让第一行的线程来完成求和（reduction）。在实际应用中，这会用更高效的并行归约算法来实现。
*   **`main` 函数**:
    1.  **设置**: 定义网络结构和超参数。
    2.  **主机数据**: 在 CPU 上创建和初始化数据。
    3.  **`cudaMalloc`**: 在 GPU 上为所有矩阵和向量分配内存。
    4.  **`cudaMemcpy`**: 将初始数据从 CPU 拷贝到 GPU。
    5.  **训练循环**:
        *   **内核启动 `<<<...>>>`**: 配置网格（Grid）和块（Block）的维度并启动内核。我们使用 2D 网格来直观地映射到矩阵。
        *   **顺序**: 严格按照前向传播 -> 反向传播 -> 更新权重的顺序启动内核。
        *   **监控**: 定期将预测结果 `d_A2` 从 GPU 拷贝回 CPU 以计算和打印损失，监控训练进程。
    6.  **`cudaDeviceSynchronize`**: 阻塞 CPU，直到 GPU 完成所有已提交的任务。
    7.  **`cudaFree`**: 释放所有在 GPU 上分配的内存。

---

### 4. 如何编译和运行

你需要安装 NVIDIA CUDA Toolkit。

```bash
# 编译
nvcc nn_cuda.cu -o nn_cuda

# 运行
./nn_cuda
```

你会看到损失值随着训练的进行而下降，最后输出对XOR问题的预测结果，表明模型已经成功学习。

### 5. 性能瓶颈与优化方向

这个 "Naive" 实现性能远非最优，主要瓶颈在于**对全局内存的低效访问**。

1.  **矩阵乘法**: 每个线程中的 `for` 循环都在反复读取全局内存。**优化方法**: 使用**共享内存（Shared Memory）**。将输入矩阵的块（Tiles）加载到快速的共享内存中，然后从共享内存进行计算，可以极大地减少全局内存访问次数，性能提升一个数量级。
2.  **使用 cuBLAS**: 对于矩阵乘法，自己手写永远比不过 NVIDIA 官方的 **cuBLAS** 库。这是一个高度优化的线性代数库，是所有深度学习框架（TensorFlow, PyTorch）的基石。在实际应用中，你会调用 `cublasSgemm()` 函数来执行矩阵乘法。
3.  **内核融合 (Kernel Fusion)**: 我们已经做了一点融合（例如 `forward_kernel` 同时计算 `Z` 和 `A`）。可以进一步融合，比如将误差计算和权重更新合并到更少的内核中，以减少内核启动的开销和全局内存的读写。
4.  **偏置更新**: `update_weights_kernel` 中更新偏置 `b` 的方式效率低下。应该使用高效的并行归约（Parallel Reduction）内核来完成求和。

尽管有这些瓶颈，但这个手写的例子完美地展示了将神经网络计算映射到 GPU 并行架构的基本思想。