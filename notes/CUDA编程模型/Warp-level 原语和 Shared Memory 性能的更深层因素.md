
---

### 微观世界的法则与宏观模式的构建

#### **第一幕：深入硬件的微观世界 - 寄存器与占用率 (Occupancy)**

我们之前讨论了 Shared Memory 的银行冲突。但还有一个更基础、更普遍的资源限制，那就是**寄存器 (Registers)** 和**占用率 (Occupancy)**。

*   **寄存器 (Registers)**:
    *   **是什么**: 这是 GPU SM 上**最快**的存储单元，每个线程都有一组私有的寄存器，用于存储其局部变量（如循环计数器、累加和等）。
    *   **限制**: 每个 SM 的寄存器文件 (Register File) 的**总容量是有限的**（例如，一个 A100 SM 有 65536 个 32-bit 寄存器）。
    *   **影响**: 如果你的 Kernel 函数中使用了大量的局部变量，或者编译器因为复杂的代码逻辑而无法有效地复用寄存器，那么**每个线程所需的寄存器数量就会增加**。

*   **占用率 (Occupancy)**:
    *   **定义**: 一个 SM 上**活跃的 Warp 数量 (Active Warps)** 与该 SM **所能支持的最大活跃 Warp 数量** 的比值。
    *   **为什么重要**: 占用率是**隐藏内存延迟的关键**。当一个 Warp 因为等待全局内存读取而停顿时（stall），如果 SM 上有其他活跃的 Warp 处于就绪状态，GPU 的 Warp 调度器就可以**立即切换**到那个就绪的 Warp，开始执行它的指令。这样，SM 的计算单元就不会空闲。**高占用率意味着调度器有更多的“备胎”Warp 可供选择，从而能更好地隐藏延迟。**
    *   **计算公式（简化版）**: `Occupancy = (实际并发 Block 数 * 每个 Block 的 Warp 数) / (每个 SM 的最大 Block 数 * 每个 SM 的最大 Warp 数)`

**寄存器与占用率的“跷跷板”关系**:

这是 CUDA 性能调优中最经典的权衡之一：
*   SM 的总寄存器数量是固定的。
*   `总寄存器数 = (每个 SM 的活跃 Warp 数) * (每个 Warp 的线程数) * (每个线程使用的寄存器数)`
*   `总寄存器数 = (每个 SM 的活跃 Warp 数) * 32 * (每个线程使用的寄存器数)`

从这个公式可以看出：
**在一个 Block Size 固定的情况下，如果你的 Kernel 让每个线程使用了过多的寄存器，那么一个 SM 能同时容纳的活跃 Warp 数量就会下降，从而导致占用率降低！**

**如何分析和优化？**
1.  **编译时获取信息**: 使用 `nvcc` 的 `-Xptxas -v` 或 `--ptxas-options=-v` 标志，编译器会告诉你每个 Kernel 使用了多少寄存器和 Shared Memory。
    ```bash
    nvcc my_kernel.cu -o my_kernel -Xptxas -v
    # ptxas info    : 0 bytes gmem
    # ptxas info    : Compiling entry function '_Z10my_kernel...' for 'sm_80'
    # ptxas info    : Function properties for _Z10my_kernel...
    #     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    # ptxas info    : Used 48 registers, 340 bytes cmem[0]
    ```
    这里显示使用了 48 个寄存器。
2.  **CUDA Occupancy Calculator**: NVIDIA 提供了一个 Excel 表格，你可以输入你的 GPU 型号、Block Size、每个线程的寄存器使用量和 Shared Memory 使用量，它会精确地计算出你的理论占用率。这是一个非常有用的调试工具。
3.  **优化策略**:
    *   **减少局部变量**: 复用变量，避免不必要的临时变量。
    *   **启动边界 (`__launch_bounds__`)**: 你可以给 Kernel 函数加上 `__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)` 限定符。这会向编译器提供一个“提示”，告诉它这个 Kernel 最多会以 `MAX_THREADS_PER_BLOCK` 的大小启动，并且你希望每个 SM 至少能跑 `MIN_BLOCKS_PER_MULTIPROCESSOR` 个 Block。编译器会尝试更积极地优化寄存器使用，以满足这个占用率目标，但有时可能会以增加指令数或引入“寄存器溢出 (Register Spilling)”（将寄存器数据存入慢速的本地内存）为代价。