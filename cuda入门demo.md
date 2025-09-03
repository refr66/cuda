好的，这三个问题——Block Reduce、矩阵转置、矩阵乘法——是CUDA并行编程的“入门三件套”。它们完美地展示了如何利用线程块（Thread Block）和共享内存（Shared Memory）来设计高效的并行算法。

我们逐一来看如何手写它们的CUDA实现。

---

### 1. Block Reduce (线程块内归约)

**目标**: 对一个线程块（Block）内的所有线程持有的数据进行求和（或其他归约操作，如求最大/最小值）。这是许多并行算法（如求全局总和、向量点积）的基础构件。

**核心思想**: **二分减半 (Divide and Conquer)**。
在一个线程块内，我们利用共享内存，通过多轮同步的“减半”操作，将计算量对数级降低。

**Naive vs. 优化实现**:

*   **Naive实现**: 线程0作为累加器，其他线程通过`atomicAdd`把自己的值加给线程0。**缺点**: 会产生严重的共享内存bank conflict（如果用原子操作）或序列化访问，效率低。
*   **优化实现 (下面展示的)**: 并行二分减半。

#### CUDA C++ 代码实现 (`block_reduce.cu`)

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel for Block-wise Reduction (Sum)
__global__ void block_reduce_kernel(const float* d_in, float* d_out, int N) {
    // 1. Allocate shared memory for this block
    // extern 关键字表示大小在内核启动时动态指定
    extern __shared__ float s_data[];

    // 2. Load data from global memory to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        s_data[tid] = d_in[i];
    } else {
        s_data[tid] = 0.0f; // Neutral element for sum
    }
    
    // Synchronize to make sure all data is loaded into shared memory
    __syncthreads();

    // 3. Perform reduction in shared memory (the core logic)
    // The loop iterates log2(blockSize) times
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Synchronize after each step to ensure writes are visible
        __syncthreads();
    }

    // 4. The first thread in the block writes the result to global memory
    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

int main() {
    const int N = 1024 * 1024; // Total number of elements
    const int BLOCK_SIZE = 256; // Threads per block

    // Calculate grid size
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate host memory
    std::vector<float> h_in(N);
    std::iota(h_in.begin(), h_in.end(), 1.0f); // Fill with 1, 2, 3, ...
    std::vector<float> h_out(GRID_SIZE);

    // Allocate device memory
    float *d_in, *d_out;
    gpuErrchk(cudaMalloc(&d_in, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_out, GRID_SIZE * sizeof(float)));

    // Copy data to device
    gpuErrchk(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    // The third argument specifies the dynamic shared memory size per block
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    block_reduce_kernel<<<GRID_SIZE, BLOCK_SIZE, shared_mem_size>>>(d_in, d_out, N);
    
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back to host
    gpuErrchk(cudaMemcpy(h_out.data(), d_out, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verification
    float total_sum = 0.0f;
    for(float val : h_out) {
        total_sum += val;
    }

    // This is a large sum, use double for CPU calculation to maintain precision
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_in[i];
    }
    
    std::cout << "GPU Reduction Sum: " << total_sum << std::endl;
    std::cout << "CPU Sum: " << cpu_sum << std::endl;
    if (abs(total_sum - cpu_sum) / cpu_sum < 1e-5) {
        std::cout << "Result is CORRECT!" << std::endl;
    } else {
        std::cout << "Result is INCORRECT!" << std::endl;
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
```

**编译与运行**:
`nvcc block_reduce.cu -o block_reduce && ./block_reduce`

**关键点**:
1.  `extern __shared__ float s_data[]`: 动态分配共享内存。大小在内核启动时通过第三个参数 `<<<...>>>` 指定。
2.  `__syncthreads()`: **至关重要**。它是一个同步屏障，确保块内所有线程都到达这一点后，才继续执行。在归约的每一步，都需要它来保证上一轮的加法结果已经写回共享内存，对所有线程可见。
3.  **循环**: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)`。`s` 是步长，每次减半。如果 `blockDim.x` 是256，`s` 依次会是128, 64, 32, 16, 8, 4, 2, 1。
4.  **最终结果**: 归约完成后，最终的总和存储在 `s_data[0]` 中，由线程0写回全局内存。

---

### 2. 矩阵转置 (Matrix Transpose)

**目标**: 将一个 `M x N` 的矩阵 `A` 转换为 `N x M` 的矩阵 `A^T`。

**Naive 实现**: 每个线程计算一个输出元素：`out[j*M + i] = in[i*N + j]`。**缺点**: 这种访问模式对于输入矩阵 `in` 是**非合并的 (non-coalesced)**，导致内存访问效率低下。一个warp中的32个线程会访问全局内存中分散的地址，无法有效利用内存带宽。

**优化实现**: **使用共享内存分块 (Tiling)**。

#### CUDA C++ 代码实现 (`transpose.cu`)

```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_DIM 16 // Tile width
#define BLOCK_ROWS 16 // Threads per block y-dim

// Kernel for tiled matrix transpose
__global__ void transpose_kernel(const float* in, float* out, int width, int height) {
    // 1. Declare shared memory tile
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    // 2. Calculate global indices
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 3. Load data from global memory to shared memory (coalesced access)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    
    // Synchronize to ensure the tile is fully loaded
    __syncthreads();

    // 4. Calculate transposed global indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 5. Write data from shared memory to global memory (coalesced access)
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y]; // The actual transpose happens here
    }
}

// ... (main function for setup, launch, verification - similar structure to above)
// In main():
// dim3 threads(TILE_DIM, BLOCK_ROWS);
// dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
// transpose_kernel<<<grid, threads>>>(d_in, d_out, width, height);
```
**(为简洁起见，省略了main函数，其结构与上一个例子类似)**

**关键点**:
1.  **分块 (Tiling)**: 我们将大矩阵划分为 `TILE_DIM x TILE_DIM` (例如 16x16) 的小块。每个线程块负责一个块的转置。
2.  **共享内存 `tile`**: 每个块都有自己的共享内存 `tile`。
3.  **合并读取 (Coalesced Read)**: 在第3步，线程块内的线程按行顺序（`y * width + x`）读取数据。这是一个合并访问，非常高效。
4.  **共享内存转置**: 在第5步，线程从共享内存 `tile` 中读取数据时，交换了 `threadIdx.x` 和 `threadIdx.y` 的位置 (`tile[threadIdx.x][threadIdx.y]`)。真正的转置发生在这个**极快**的片上内存中。
5.  **合并写入 (Coalesced Write)**: 写入到输出矩阵 `out` 时，也是按行顺序的，同样是合并访问。
6.  `+ 1` **避免 Bank Conflict**: `__shared__ float tile[TILE_DIM][TILE_DIM + 1]`。当一个warp中的多个线程同时访问同一个共享内存bank的不同地址时，访问是并行的。但如果访问同一个bank的同一个地址，就会发生bank conflict，访问会被序列化。在转置时，列访问 `tile[threadIdx.x][threadIdx.y]` 会导致bank conflict。通过增加一列 `+1`，可以改变内存布局，错开bank，从而避免冲突。这是一个经典的优化技巧。

---

### 3. 简单的矩阵乘法 (Matrix Multiplication)

**目标**: 计算 `C = A * B`，其中 `A` 是 `M x K`，`B` 是 `K x N`，`C` 是 `M x N`。

**Naive 实现**: 每个线程计算输出矩阵 `C` 的一个元素 `C(i, j)`。该线程需要循环 `K` 次，读取 `A` 的第 `i` 行和 `B` 的第 `j` 列进行点积。**缺点**: 对矩阵 `A` 和 `B` 的全局内存访问非常频繁且冗余。`A` 的同一行被计算 `C` 同一行的所有线程重复读取；`B` 的同一列被计算 `C` 同一列的所有线程重复读取。

**优化实现**: **使用共享内存分块 (Tiling)**。

#### CUDA C++ 代码实现 (`matmul.cu`)

```cpp
#include <iostream>
// ... includes and error check macro

#define TILE_WIDTH 16

// Kernel for tiled matrix multiplication C = A * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. Declare shared memory tiles for A and B
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // 2. Thread indices
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float C_value = 0.0f;

    // 3. Loop over the tiles of A and B required to compute the C element
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        
        // --- Load one tile of A and B into shared memory ---
        // Load A's tile
        if (row < M && (t * TILE_WIDTH + tx) < K) {
            ds_A[ty][tx] = A[row * K + (t * TILE_WIDTH + tx)];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        // Load B's tile
        if (col < N && (t * TILE_WIDTH + ty) < K) {
            ds_B[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // --- Multiply the two tiles from shared memory ---
        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += ds_A[ty][k] * ds_B[k][tx];
        }
        
        // Synchronize to make sure all calculations are done before loading next tile
        __syncthreads();
    }

    // 4. Write the final result to C
    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}
// ... (main function for setup, launch, verification)
// In main():
// dim3 threads(TILE_WIDTH, TILE_WIDTH);
// dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
// matmul_kernel<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
```

**关键点**:
1.  **输出驱动**: 每个线程仍然负责计算 `C` 矩阵中的一个元素。
2.  **分块计算**: 计算 `C(row, col)` 的点积被分解为一系列小块的点积。
3.  **协同加载**: `for` 循环的每次迭代，整个线程块协同地将 `A` 的一个 `TILE_WIDTH x TILE_WIDTH` 块和 `B` 的一个 `TILE_WIDTH x TILE_WIDTH` 块从全局内存加载到共享内存 `ds_A` 和 `ds_B` 中。这是一个合并访问。
4.  **从共享内存计算**: 加载完成后，所有线程并行地从**快速的共享内存**中读取数据，计算部分点积。这极大地减少了对慢速全局内存的访问。
5.  **循环**: `for (int t ...)` 循环遍历所有需要的砖块，将部分点积累加到 `C_value` 中。
6.  **最终写入**: 循环结束后，每个线程将最终的 `C_value` 写回全局内存。

这三个例子是CUDA优化的基石。理解并能手写它们，你就掌握了CUDA编程的核心思想：**识别并行性、利用内存层次（尤其是共享内存）来减少全局内存访问、确保线程同步**。