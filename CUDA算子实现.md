
### **简单算子赛题 (T1-1)**

这个级别的算子主要考察基础的计算逻辑实现和对框架基本要求的理解，如数据类型、内存操作等。它们通常是**Element-wise（逐元素）**操作，意味着输出张量的每个元素只依赖于输入张量对应位置的一个或多个元素，不涉及复杂的维度变换和数据依赖。

#### **T1-1-1：单目 / 三目算子**

*   **算子列表**: `Exp`, `Sin`, `Cos`, `LeakyReLU`, `Tanh`, `Sigmoid Backward`, `HardSwish`, `Cast`, `Where`
*   **核心考察点**:
    1.  **数学函数实现**: 正确实现 `exp(x)`, `sin(x)`, `cos(x)`, `tanh(x)` 等基本数学函数。
    2.  **激活函数**: 实现 `LeakyReLU` (`max(0, x) + negative_slope * min(0, x)`), `HardSwish` (`x * ReLU6(x + 3) / 6`) 等现代激活函数。
    3.  **反向传播（Backward）**: `Sigmoid Backward` 是第一个反向算子，需要理解其梯度计算公式：`grad_output * (sigmoid(x) * (1 - sigmoid(x)))`。
    4.  **数据类型转换**: `Cast` 算子，实现如 `float32` 到 `float16` 的转换。
    5.  **条件选择**: `Where` 是一个三目算子，`out = condition ? x : y`，考察根据一个布尔张量选择不同来源数据的能力。
*   **特殊要求解析**:
    *   `输入输出类型一致`: 除 `Cast` 外，所有算子计算结果的数据类型应与输入一致。
    *   `支持 Inplace 操作`: "Inplace"（原地操作）指直接在输入张量的内存上修改得到输出，从而节省内存。例如，`x.exp_()` 会修改 `x` 本身，而不是返回一个新的张量。`Cast` 无法原地操作，因为改变类型可能会改变元素占用的字节数。

---

#### **T1-1-2：单目 / 双目算子**

*   **算子列表**: `Silu`, `Div`, `And`, `Or`, `Equal`, `ReLU Backward`, `GeLU`, `GeLU Backward`, `CrossEntropyLoss Backward`
*   **核心考察点**:
    1.  **双目（Binary）运算**: `Div`（除法）、`And`/`Or`（逻辑运算）、`Equal`（比较运算）需要处理两个输入张量，可能涉及 **Broadcasting（广播）机制**。
    2.  **更多激活函数及其反向**:
        *   `Silu` (或称 Swish): `x * sigmoid(x)`。
        *   `GeLU`: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` (近似公式) 或使用 `erf` 函数。
        *   `ReLU Backward`: 梯度为 `grad_output * (x > 0 ? 1 : 0)`。
        *   `GeLU Backward`: 需要实现 `GeLU` 的导数公式，比 `ReLU` 复杂。
    3.  **核心损失函数反向**: `CrossEntropyLoss Backward` 是一个非常关键的算子。它计算的是交叉熵损失相对于模型输出（logits）的梯度。其计算通常是 `softmax(logits) - one_hot_labels`，是分类任务反向传播的起点。

---

### **中等算子赛题 (T1-2)**

这个级别开始引入更复杂的计算模式，不再是简单的逐元素操作。涉及**维度规约（Reduction）**、**数据索引与重排**、**归一化**以及更复杂的**复合运算**。

#### **T1-2-1：规约与归一化**

*   **算子列表**: `ReduceMax`, `ReduceMean`, `BatchNorm` (及反向), `LayerNorm` (及反向), `RMSNorm Backward`
*   **核心考察点**:
    1.  **Reduction 操作**: `ReduceMax/Mean` 需要沿着指定的维度（`axis`）进行计算，例如对一个 `(N, C, H, W)` 的张量在 `H, W` 维度上求均值。这需要高效的并行规约算法。
    2.  **归一化算法**:
        *   `BatchNorm`: 沿 **Batch 维度**进行归一化，需要计算 mini-batch 的均值和方差。反向传播逻辑复杂，涉及链式法则的多次应用。
        *   `LayerNorm`: 沿 **Feature 维度**进行归一化，常用于 Transformer。其反向传播也需要仔细推导。
        *   `RMSNorm Backward`: `RMSNorm` 是 `LayerNorm` 的简化版，反向逻辑也相应简化。
*   **特殊要求解析**:
    *   `支持指定维度计算`: 实现必须是通用的，能处理任意给定的规约/归一化维度。
    *   `3D 输入及连续张量`: 表明测试用例至少包含 3D 张量，并且可以假设输入张量的内存是连续的，这简化了内存访问逻辑。

---

#### **T1-2-2：索引与线性变换**

*   **算子列表**: `IndexCopyInplace`, `Gather`, `Scatter`, `tril`, `triu`, `Linear` (及反向)
*   **核心考察点**:
    1.  **数据移动与索引**:
        *   `Gather`: 根据索引（`index`）从输入（`input`）中抽取数据。`output[i][j] = input[index[i][j]][j]` (示例)。
        *   `Scatter`: `Gather` 的逆操作，将源（`src`）数据根据索引（`index`）写入到目标张量中。
        *   `IndexCopyInplace`: 在指定维度上，根据索引将源数据拷贝到自身。
    2.  **结构化操作**: `tril`/`triu` 用于提取矩阵的下三角/上三角部分，常用于构建注意力掩码。
    3.  **核心网络层**: `Linear` 层 (即全连接层) 本质是**矩阵乘法** (`Y = X @ W.T + b`)。其反向需要计算对输入、权重和偏置的梯度，同样涉及矩阵乘法。
*   **特殊要求解析**:
    *   `支持任意步长 (stride)`: 这是一个重要难点。意味着不能假设张量数据在内存中是连续的，必须通过 `shape` 和 `stride` 来计算元素的实际内存地址，增加了实现的复杂性。
    *   `2D 连续张量`: 针对 `Linear` 等算子，给出了一个简化的场景，方便基础功能的实现与测试。
    *   `可选 bias`: `Linear` 层的偏置 `b` 是可选的，实现时需要处理 `bias` 存在或为 `None` 的情况。

---

#### **T1-2-3：卷积、池化与损失**

*   **算子列表**: `CrossEntropyLoss`, `AveragePool` (及反向), `MaxPool` (及反向), `InterpolateNearest`, `Conv Backward`
*   **核心考察点**:
    1.  **损失函数前向**: `CrossEntropyLoss` 的正向计算，通常结合了 `LogSoftmax` 和 `NLLLoss`。
    2.  **池化操作**: `AvgPool` 和 `MaxPool` 是 CNN 的关键组件，用于下采样。它们的反向传播逻辑不同：`AvgPool` 的梯度被平均分配给感受野内的所有元素；`MaxPool` 的梯度只回传给取得最大值的那个元素。
    3.  **插值/上采样**: `InterpolateNearest`（最近邻插值）是一种简单的上采样方法。
    4.  **卷积反向**: `Conv Backward` 是这个级别中最复杂的算子之一。需要计算三个梯度：
        *   `grad_input` (对输入的梯度): 通常通过**转置卷积 (Transposed Convolution)** 实现。
        *   `grad_weight` (对权重的梯度): 通常通过对输入进行卷积来实现。
        *   `grad_bias` (对偏置的梯度): 对输出梯度在相应维度上求和。
*   **特殊要求解析**:
    *   `支持 1D-3D 场景`: 要求池化、卷积等操作不仅支持 2D 图像，还要能处理 1D 序列和 3D 体数据。
    *   `精度验证`: 强调了与 PyTorch 对标的重要性，需要保证数值计算的准确性。

---

### **困难算子赛题 (T1-3)**

这个级别是关于当前（SOTA）大模型中的核心、高性能算子。它们不仅算法复杂，而且对性能有极致要求，通常需要深入理解底层硬件架构（如 GPU）才能实现得好。

#### **T1-3-1：FlashAttention**

*   **算子列表**: `FlashAttention` (及反向)
*   **核心考察点**:
    1.  **Attention 机制理解**: 深刻理解标准 Self-Attention 的计算瓶颈——即 `(N, N)` 大小的注意力矩阵的显存占用和读写开销。
    2.  **FlashAttention 算法**: 实现 FlashAttention 论文中提出的核心思想：**Tiling（分块）** 和 **Online Softmax**。通过将 Q, K, V 矩阵分块，在 GPU 的片上高速缓存（SRAM）中完成子块的矩阵乘法和 Softmax 计算，避免将巨大的中间结果写入主显存（HBM），从而大幅减少 I/O，实现加速。
    3.  **反向传播**: FlashAttention 的反向传播同样利用了 Tiling 和重计算（recomputation）技巧，需要重新推导梯度并以分块方式实现。
*   **特殊要求解析**:
    *   `支持三种 mask`:
        *   `无 mask`: 用于 Encoder-only 模型如 BERT。
        *   `causal mask` (因果掩码): 用于 Decoder-only 模型如 GPT，确保每个 token 只能注意到它之前的 tokens。
        *   `自传 mask`: 支持一个外部传入的任意布尔掩码。
    *   `性能优于标准 attention`: 这是硬性指标。实现必须比 `softmax(Q @ K.T) @ V` 的朴素实现快得多，特别是在长序列场景下。

---

#### **T1-3-2：前沿模型定制算子**

*   **算子列表**: `Latent Attention Multi-head`, `Top-K Router`
*   **核心考察点**:
    1.  **论文/模型复现能力**: 这类算子通常没有标准库，需要阅读指定模型（如 Deepseek V3）的论文或源代码，理解其独特的计算逻辑。
    2.  **`Latent Attention Multi-head`**: 可能是一种变种的注意力机制。例如，它可能采用了 Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)，或者有特殊的 KV Cache 处理方式，或者是像 "latent" 所暗示的，对某些 "潜在" 的压缩表示进行 attention。
    3.  **`Top-K Router`**: 这是 **Mixture-of-Experts (MoE)** 架构的核心。一个路由网络（Router）会为每个 token 计算一个得分，然后选择得分最高的 Top-K 个专家（Expert）来处理该 token。这个算子的实现涉及：
        *   计算路由权重。
        *   执行 Top-K 选择。
        *   根据选择结果，将 token 分发（dispatch）到不同的专家，这通常是一个复杂的 `gather`/`scatter` 操作。
*   **特殊要求解析**:
    *   `动态 cache 位置`: 在大模型推理中，KV Cache 的管理至关重要。`动态位置` 可能指 PagedAttention 这类先进的 KV Cache 管理技术，它将 Cache 分成非连续的块，允许更灵活的内存管理和更高的吞吐量。
    *   `指定参数格式`: 高性能推理框架通常会对权重等参数进行重排（reorder）以优化内存访问模式。实现需要适配这种特定的数据布局。

### **总结**

这个赛题设计得非常全面，覆盖了从深度学习基础到前沿大模型优化的全链路能力：

*   **简单级**: 考验**基础编程**和**数学知识**。
*   **中等级**: 考验对**核心 DL 算法**（归一化、卷积、索引）的深入理解和**通用并行计算模式**（如规约）的实现能力。
*   **困难级**: 考验对**SOTA 算法**的理解、**高性能计算**知识（如 GPU 内存层次结构、I/O 优化）以及**前沿模型架构**的复现能力。

要完成这些赛题，不仅需要扎实的编程功底，还需要深入的深度学习理论知识和一定的系统及硬件架构知识。