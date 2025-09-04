1.  **学习GPU微架构:**
    *   选择一个具体的架构，比如Ampere (A100) 或 Hopper (H100)，阅读NVIDIA官方发布的架构白皮书（Whitepaper）。
    *   **SM (Streaming Multiprocessor):** 它是GPU的“心脏”。理解SM的构成：
        *   **CUDA Cores:** 真正执行计算的单元（主要是FP32/INT32）。
        *   **Tensor Cores:** 用于加速矩阵运算的专用硬件。
        *   **Load/Store Units (LD/ST):** 负责访存的单元。
        *   **Special Function Units (SFU):** 用于执行`sin`, `cos`, `log`等特殊函数。
        *   **Register File & Shared Memory:** SM内部的存储资源。
    *   **内存子系统:** L1/L2缓存、显存控制器等。

2.  **连接代码与硬件:**
    *   思考一个Block是如何被调度到SM上的。
    *   思考一个Warp的指令是如何在SM的计算单元上执行的。
    *   当你的代码访问`__shared__`内存时，数据流是如何在SM内部流动的。
    *   当你调用`mma.sync` (Tensor Core指令) 时，硬件到底做了什么。