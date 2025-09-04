
3.  **学习PTX和SASS:**
    *   **PTX (Parallel Thread eXecution):** 了解这个虚拟指令集，它是连接高级语言和硬件的桥梁。尝试用 `nvcc -ptx` 编译你的代码，看看生成的PTX是什么样子。
    *   **SASS (Streaming Assembler):** 这是真正的硬件机器码。通过 `cuobjdump -sass` 可以查看。虽然不需要手写SASS，但能看懂一些基本的SASS指令，可以让你对JIT编译器的优化、寄存器使用情况有更直观的认识。