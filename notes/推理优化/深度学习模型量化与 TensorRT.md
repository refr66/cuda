好的，宗师之路已经开启！作为 Tensor Core 实战大师，我将带你深入理解 **深度学习模型量化 (INT8/FP8) 与 NVIDIA TensorRT**。这将是你部署高性能、低延迟深度学习推理模型的必修课，也是将 Tensor Core 潜力发挥到极致的关键领域。

我们将严格按照你指定的学习路径，逐一详细讲解。

---

### **引言：为什么量化是推理的“核武器”？**

在模型训练阶段，我们通常使用 FP32 或混合精度 (FP16/TF32) 来保证数值稳定性和收敛性。然而，当模型训练完成，进入**推理 (Inference)** 阶段时，我们对性能和资源的需求会发生变化：

*   **极致低延迟：** 用户希望请求能立即得到响应。
*   **高吞吐量：** 每秒处理更多请求。
*   **低内存占用：** 部署到边缘设备、IoT 设备，或在数据中心中节省昂贵的显存。
*   **低功耗：** 延长电池寿命，或降低数据中心运营成本。

浮点数运算，尤其是 FP32，在这方面表现不佳：它们需要更多的内存来存储，更多的计算周期来处理。而 **整数运算 (INT8)** 或 **低位浮点数 (FP8)**，由于其数据表示更紧凑，运算指令更简单，成为推理阶段的“核武器”。Tensor Core 正是为高效执行这些低精度运算而生。

**量化 (Quantization)**，简单来说，就是将模型权重和/或激活值从高精度浮点数（如 FP32）转换为低精度表示（如 INT8 或 FP8）的过程。

### **1. 量化基础：从浮点到整数的映射**

#### **1.1 核心概念：映射与缩放**

量化是将一个连续的浮点数值范围，映射到有限的离散整数集合的过程。这个映射通常涉及一个**缩放因子 (Scale Factor)** 和一个**零点 (Zero Point)**。

*   **Scale (S)：** 决定了每个整数步长代表的浮点数值范围。`S = (max_float - min_float) / (max_int - min_int)`。
*   **Zero Point (Z)：** 浮点数 `0.0` 对应的整数值。这对于处理包含负值的浮点数尤其重要。
    *   `q = round(f / S) + Z` (量化公式)
    *   `f = (q - Z) * S` (反量化公式)
    *   其中 `f` 是浮点数，`q` 是量化后的整数。

#### **1.2 量化类型 (Mapping Schemes)**

量化方案的选择会影响精度和实现复杂性。

*   **A. 对称量化 (Symmetric Quantization)：**
    *   将浮点数范围对称地映射到整数范围（例如 `[-X, +X]` 映射到 `[-127, +127]`）。
    *   零点通常是 `0` 或 `0` 附近。
    *   优点：实现简单，尤其在硬件中执行乘法时。
    *   缺点：如果浮点数分布不是完美的对称且包含很多负值，可能无法充分利用整数的表示范围。

*   **B. 非对称量化 (Asymmetric Quantization) / 仿射量化 (Affine Quantization)：**
    *   将浮点数范围映射到整数范围时，不强制对称（例如 `[min_float, max_float]` 映射到 `[0, 255]` 或 `[-128, 127]`）。
    *   引入了非零的 `Zero Point`。
    *   优点：更灵活，能更好地适应不规则的浮点数分布，减少量化误差。
    *   缺点：实现更复杂，特别是对于零点的处理会增加计算开销。

**实战大师提示：**
*   **TensorRT 主要使用非对称量化。** 在 TensorRT 中，输入激活通常量化到 `[0, 255]` (无符号 INT8)，权重通常量化到 `[-127, 127]` (有符号 INT8)。
*   **激活值：** 通常使用无符号 INT8 (`[0, 255]`)，因为 ReLU 激活函数输出非负值。
*   **权重：** 通常使用有符号 INT8 (`[-127, 127]`)，因为权重可以是负数。

#### **1.3 量化数据类型 (INT8, FP8)**

*   **INT8 (8-bit Integer)：**
    *   **表示范围：** 有符号 `[-128, 127]`，无符号 `[0, 255]`。
    *   **优势：**
        *   **极致小巧：** 1/4 于 FP32，1/2 于 FP16 的内存占用。
        *   **极致快速：** 整数运算速度快，且 Tensor Core 对 INT8 有原生支持，提供极高吞吐量。
    *   **挑战：**
        *   **精度损失：** 只有 256 个离散值，量化误差大。
        *   **动态范围有限：** 如果浮点数范围过大，需要进行“裁剪” (clipping)，超出范围的值会被映射到最大/最小值，导致精度严重损失。
        *   **需要校准/训练：** 找到合适的 `Scale` 和 `Zero Point` 是关键。

*   **FP8 (8-bit Floating Point)：**
    *   **背景：** NVIDIA 在 Hopper (H100) 架构中引入，旨在解决超大规模模型训练和推理中的内存和计算瓶颈。
    *   **表示：** 有两种主要格式 (NVIDIA/Onyx)：
        *   **E4M3 (4位指数，3位尾数)：** 动态范围较小，但精度较高。适合表示权重和激活。
        *   **E5M2 (5位指数，2位尾数)：** 动态范围较大，但精度较低。适合表示梯度。
    *   **优势：**
        *   **比 INT8 更大的动态范围：** 浮点数特性，减少了溢出/下溢的风险，无需像 INT8 那样频繁进行裁剪。
        *   **比 INT8 更好的精度 (在某些情况下)：** 对于某些难以量化为 INT8 的模型，FP8 可能提供更好的精度-性能平衡。
        *   **Tensor Core 原生支持：** H100 的 Tensor Core 对 FP8 进行了高度优化，提供了前所未有的计算吞吐量。
    *   **挑战：**
        *   **仍在发展中：** 标准和最佳实践仍在演进。
        *   **并非所有 GPU 都支持：** 主要在 Hopper 及更高级别的架构上。
        *   **复杂性：** 需要动态精度选择、混合精度训练的更多技巧。

#### **1.4 量化误差**

量化不可避免地会引入误差，主要包括：

*   **截断误差 (Rounding Error)：** 浮点数转换为离散整数时，会进行四舍五入。
*   **裁剪误差 (Clipping Error) / 溢出误差：** 当浮点数超出量化范围时，会被强制映射到最大或最小整数值，导致信息丢失。

**实战大师提示：** 量化的核心挑战在于如何最小化这些误差，从而在保持模型准确率的同时获得性能提升。

### **2. 量化技术：何时以及如何量化**

根据量化发生的时间和是否需要模型重新训练，量化技术分为两大类。

#### **2.1 训练后量化 (Post-Training Quantization, PTQ)**

*   **定义：** 模型在 FP32 精度下训练完成后，在不重新训练或微调的情况下，将权重和/或激活量化到低精度。
*   **流程：**
    1.  **模型训练：** 在 FP32 或混合精度下完成模型训练。
    2.  **收集统计信息 (Calibration / 校准)：**
        *   为了确定最佳的 `Scale` 和 `Zero Point`，需要使用一小部分**有代表性**的未标注数据（校准集）运行模型的前向传播。
        *   在此过程中，收集每层权重和激活值的统计信息，如 `min/max` 值分布或更复杂的统计（如 KL 散度）。
    3.  **量化转换：** 根据收集到的统计信息，计算 `Scale` 和 `Zero Point`，然后将模型权重和激活图谱转换为低精度。
*   **优点：**
    *   **简单快捷：** 无需重新训练，成本低。
    *   **部署友好：** 直接在训练好的模型上操作。
*   **缺点：**
    *   **精度损失可能较大：** 对于复杂模型或对精度敏感的模型，PTQ 可能会导致显著的准确率下降。
    *   **对校准数据敏感：** 校准集必须充分代表实际推理数据分布。
*   **适用场景：** 对精度损失不敏感的模型；快速原型验证；资源受限的边缘设备（精度要求不高）。

#### **2.2 量化感知训练 (Quantization-Aware Training, QAT)**

*   **定义：** 在训练过程中引入“假量化” (Fake Quantization) 操作。这意味着模型在训练时，虽然权重和激活仍然是浮点数，但它们会模拟量化操作的影响，让模型“感知”到量化引起的误差，并学习如何补偿这些误差。
*   **流程：**
    1.  **插入假量化模块：** 在模型中（通常是卷积和全连接层之后）插入模拟量化和反量化的操作。这些操作在训练时会将浮点数四舍五入到最近的量化值，然后立即反量化回浮点数，进行后续计算。
    2.  **微调 (Fine-tuning)：** 使用一个小的学习率，在包含假量化模块的模型上进行额外的训练（通常是微调已预训练好的模型）。
    3.  **最终量化：** 微调完成后，移除假量化模块，直接将权重和激活量化为 INT8/FP8。
*   **优点：**
    *   **更高的准确率：** 模型在训练过程中适应量化误差，通常能保持接近 FP32 的准确率。
    *   **更鲁棒：** 对不同的量化方案和数据分布有更好的适应性。
*   **缺点：**
    *   **训练成本高：** 需要重新训练或微调模型。
    *   **实现复杂：** 需要修改模型结构或使用框架提供的 QAT API。
*   **适用场景：** 对模型精度要求高的任务；当 PTQ 无法达到满意准确率时。

**实战大师提示：** QAT 通常是获得最佳 INT8 推理性能和准确率平衡的方法。

### **3. 框架特定量化工具 (PyTorch & TensorFlow)**

主流深度学习框架都提供了对量化操作的支持，它们是进行 PTQ 或 QAT 的起点。

#### **3.1 PyTorch Quantization API**

PyTorch 提供了 `torch.quantization` 模块，支持多种量化策略：

*   **Eager Mode Quantization (PTQ)：** 允许你在模型训练完成后，通过几行代码对模型进行量化。你需要定义一个量化配置 (`qconfig`)，然后使用 `torch.quantization.prepare` 和 `torch.quantization.convert`。
    ```python
    import torch.quantization
    
    # 示例: PTQ 静态量化 (针对权重和激活)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm') # 'qnnpack' for ARM CPU, 'fbgemm' for X86 CPU
    # 准备模型，插入观察者 (Observer) 来收集激活值的统计信息
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32_fused, [['conv1', 'relu1'], ['conv2', 'relu2'], ['fc1', 'relu3']])
    model_prepared = torch.quantization.prepare(model_fp32_fused)
    
    # 在校准数据集上运行前向传播
    # for data, target in calib_loader:
    #    model_prepared(data)
    
    # 转换模型为量化版本
    model_quantized = torch.quantization.convert(model_prepared)
    ```
*   **Quantization-Aware Training (QAT)：** PyTorch 也提供了在训练循环中启用 QAT 的方法。
    ```python
    # 示例: QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_prepared_qat = torch.quantization.prepare_qat(model_fp32_fused)
    # 在模型上运行正常的训练循环 (微调)
    # for epoch in range(num_epochs):
    #    train_one_epoch(model_prepared_qat, ...)
    # 训练结束后，转换模型
    model_quantized_qat = torch.quantization.convert(model_prepared_qat.eval())
    ```

#### **3.2 TensorFlow Lite Quantization**

TensorFlow 提供了 TensorFlow Lite Converter 来进行模型优化和量化，主要目标是移动和边缘设备。

*   **PTQ (Post-Training Quantization)：**
    *   **Dynamic Range Quantization：** 仅量化权重到 INT8，激活在运行时根据动态范围进行量化。
    *   **Full Integer Quantization：** 权重和激活都量化到 INT8。需要一个代表性数据集进行校准。
*   **QAT (Quantization-Aware Training)：** TensorFlow Model Optimization Toolkit 提供了 QAT API。

**实战大师提示：** 框架自带的量化工具通常用于生成可以在 CPU 上运行的量化模型，或者作为导出到 TensorRT 的中间步骤。要充分利用 GPU 上的 Tensor Core 进行 INT8/FP8 推理，**TensorRT 才是你的首选。**

### **4. NVIDIA TensorRT：推理部署利器**

TensorRT 是 NVIDIA 针对高性能深度学习推理的 SDK。它是一个用于优化神经网络的编程库，能够自动对模型进行多项优化，其中就包括对 Tensor Core 友好的低精度量化。

#### **4.1 TensorRT 的核心功能**

*   **层融合 (Layer Fusion)：** 将多个连续的层（如 Conv + ReLU + BatchNorm）融合成一个单独的 Kernel，减少内存访问和 Kernel 启动开销。
*   **精度优化：** 支持 FP32, FP16, INT8, FP8 精度，并自动选择最优的 Tensor Core Kernel。
*   **内存优化：** 减少显存占用，复用内存。
*   **Kernel 自动调优：** 为目标 GPU 架构选择最佳的算法和 Kernel。

#### **4.2 TensorRT 的核心组件**

TensorRT 工作流通常包括以下几个步骤：

1.  **模型解析 (Parser)：**
    *   **作用：** 将你的深度学习模型从各种框架格式（如 ONNX, UFF, PyTorch TorchScript）转换为 TensorRT 内部的 IR (Intermediate Representation)。
    *   **主要 Parser：**
        *   `ONNX Parser` (推荐)：将 ONNX 格式的模型导入 TensorRT。ONNX 是一种开放的神经网络交换格式，几乎所有主流框架都支持导出为 ONNX。
        *   `UFF Parser` (旧版，逐渐弃用)：用于从 TensorFlow 导入模型。
        *   直接从 PyTorch `torch.jit.trace` 或 `torch.jit.script` 导出的 TorchScript 模型也可以被 TensorRT 直接处理（通常通过 ONNX 路径）。
    *   **实战大师提示：** **ONNX 是最佳实践。** 确保你的模型能够正确地导出为 ONNX 格式，因为它是 TensorRT 最稳定和功能最丰富的输入格式。

2.  **构建优化引擎 (Builder)：**
    *   **作用：** 这是 TensorRT 的“大脑”，它会根据你指定的配置（如批次大小、精度模式、工作空间限制等）对解析后的网络进行一系列复杂的图优化和 Kernel 选择，最终生成一个高度优化的推理引擎。
    *   **精度配置：** 在 Builder 阶段，你可以指定是使用 FP32、FP16 还是 INT8 精度。
        *   **INT8 模式下：** Builder 会执行 INT8 校准流程（如果需要），并决定每层如何量化以及选择哪个 Tensor Core INT8 Kernel。
        *   **FP8 模式下 (H100+):** Builder 会利用 Transformer Engine 相关的优化。
    *   **工作空间 (Workspace)：** Builder 需要一定的 GPU 显存作为临时工作空间来执行优化算法。

3.  **推理引擎 (Engine)：**
    *   **作用：** Builder 的输出是一个序列化的 `.plan` 文件（或 `.trt` 文件），这就是高度优化的 TensorRT 推理引擎。它包含了所有优化后的层、量化参数和特定于 GPU 的 Kernel。
    *   **部署：** 这个引擎可以在任何支持 TensorRT 的 NVIDIA GPU 上加载并直接用于推理，而无需依赖原始深度学习框架。
    *   **上下文 (Context)：** 在运行时，你需要为引擎创建一个执行上下文 (`IExecutionContext`)，所有推理都在这个上下文中进行。

#### **4.3 INT8 校准在 TensorRT 中的实现**

对于 INT8 PTQ，TensorRT 需要一个校准过程来确定每层激活的 `Scale` 和 `Zero Point`。

*   **校准器的作用：** `IInt8Calibrator` 接口是 TensorRT 进行 INT8 校准的核心。你需要实现这个接口，提供一个批次的**代表性数据**给 TensorRT。
*   **校准过程：**
    1.  你提供一个 `batch_stream`（一个迭代器，每次返回一个批次的输入数据）。
    2.  TensorRT 遍历这个 `batch_stream`，将数据输入模型，在每层激活输出处收集统计信息（通常是直方图）。
    3.  根据这些统计信息和预设的算法（如 KL 散度最小化），计算出最优的 `Scale` 和 `Zero Point`。
    4.  这些校准结果会存储在一个缓存文件（例如 `calibration.cache`）中，以便下次直接加载，避免重复校准。
*   **校准器类型：**
    *   `IInt8EntropyCalibrator2` (推荐)：使用 KL 散度来优化 `Scale` 和 `Zero Point`，旨在最小化信息损失。
    *   `IInt8LegacyCalibrator` (旧版)：基于最大绝对值校准，效果可能不如 `EntropyCalibrator2`。
*   **实战大师提示：**
    *   **校准数据集的质量至关重要！** 它必须涵盖你模型实际推理时可能遇到的各种输入数据分布。如果校准数据不具代表性，量化后的精度会显著下降。
    *   **校准速度：** 校准过程是 CPU 密集型的，可能需要一些时间。但它是 PTQ INT8 模型的关键一步。

#### **4.4 FP8 量化与 Transformer Engine (Hopper H100)**

*   **为 Transformer 优化：** H100 GPU 上的 Tensor Core 引入了 FP8 精度，并通过 **Transformer Engine** 进一步优化了大型 Transformer 模型的训练和推理。
*   **动态精度选择：** Transformer Engine 并非简单地将所有计算都强制为 FP8。它会根据张量的数值动态范围，在 FP8 (E4M3), FP8 (E5M2) 和 FP16 之间智能切换，以在保持数值稳定性的同时，最大化 Tensor Core 的利用率。
*   **使用方式：** 通常通过 NVIDIA 提供的框架集成或库来启用，例如 PyTorch 的 `nvfuser` 或专门的 Transformer 库。开发者无需直接处理 FP8 数据类型，而是由底层框架和 TensorRT 自动管理。
*   **挑战：** FP8 更加前沿，通常需要最新的硬件和软件栈。对于大多数传统 CNN 或小型 NLP 模型，FP16 或 INT8 仍是更主流且成熟的选择。

### **5. 实践考量与最佳实践**

*   **精度与性能的权衡：** 量化总是性能与精度之间的权衡。在追求极致性能时，要时刻关注模型准确率是否在可接受范围内。
*   **验证：** 量化后务必在验证集上严格评估模型准确率，确保没有显著下降。
*   **模型结构：** 某些模型结构（如包含 Softmax、LayerNorm 等）可能对量化更敏感。这些层在 TensorRT 内部可能会被保留在 FP32/FP16 精度，以避免精度损失。
*   **调试：**
    *   **精度下降：** 首先检查校准数据集是否具有代表性。尝试使用 QAT。
    *   **TensorRT 失败：** 检查 ONNX 导出是否正确，模型是否包含 TensorRT 不支持的操作。使用 `trtexec --verbose` 或 TensorRT Logger 详细查看错误信息。
*   **INT8 Tensors：** 在 TensorRT 内部，数据以 INT8 形式流转，但你需要记住它们代表的是经过 `Scale` 和 `Zero Point` 映射的浮点数。
*   **混合精度推理：** 即使你选择 INT8 推理，TensorRT 也不会将所有操作都强制为 INT8。它会智能地选择最佳精度：计算量大的核心操作（如矩阵乘法）用 INT8/FP8 Tensor Core，而对精度敏感的操作则可能用 FP16 或 FP32。

### **总结与展望**

在本讲中，我们全面深入地探讨了深度学习模型量化的原理、关键技术（PTQ, QAT）以及它们如何与 Tensor Core 紧密结合。我们详细剖析了 NVIDIA TensorRT 作为推理优化和部署利器的核心功能和工作流程，特别是其在 INT8 校准方面的实现。最后，我们展望了 FP8 在 Hopper 架构下的最新进展。

**实战大师的思考：** 量化是实现高性能、低延迟 AI 推理的必然选择。熟练掌握 TensorRT，将你的 FP32/FP16 模型转化为高效的 INT8/FP8 推理引擎，是你在部署生产级 AI 应用中的核心竞争力。记住，量化是艺术与科学的结合，需要你不断地实验、校准和验证。

接下来，我们将继续深入其他的宗师之路学习点，或者如果你有兴趣，我们可以专门进行一个 TensorRT 和 INT8 量化的**实战演练**，从模型导出到 TensorRT 引擎构建，并进行性能和精度对比。

敬请期待！如果你有任何疑问，欢迎随时提出。


太棒了！我们现在将进行一次深度学习模型量化 (INT8) 与 NVIDIA TensorRT 的实战演练。这将是一次从模型训练、导出、量化到部署推理的完整旅程。

**本次实战的目标：**
1.  训练一个简单的 FP32 CNN 模型。
2.  将训练好的模型导出为 ONNX 格式。
3.  使用 TensorRT 分别构建 FP32 和 INT8 推理引擎。
4.  对比 PyTorch FP32、TensorRT FP32 和 TensorRT INT8 引擎的性能（推理时间）和精度。
5.  深入理解 INT8 量化校准在 TensorRT 中的作用。

---

## **Tensor Core实战大师：宗师之路 (第七讲：TensorRT INT8 量化实战演练)**

### **环境准备**

在开始之前，请确保你的环境已配置好：

1.  **NVIDIA GPU：** 支持 Tensor Core 的 GPU (Ampere/Turing/Volta 架构及以上，尤其对于 INT8)。
2.  **CUDA Toolkit：** 已安装。
3.  **PyTorch：** 已安装，且支持 CUDA。
4.  **TensorRT：** **这是最关键的一步。** TensorRT 通常作为独立安装包或通过 Docker 容器提供。
    *   **推荐使用 Docker：** 最简单、最可靠的方式是使用 NVIDIA 提供的 NGC TensorRT 容器。
        ```bash
        # 拉取最新 TensorRT 容器 (例如 23.08-py3, 对应TRT 8.6)
        docker pull nvcr.io/nvidia/tensorrt:23.08-py3
        
        # 运行容器，挂载当前目录以便访问代码和数据
        docker run --gpus all -it --rm -v $(pwd):/workspace/host_code nvcr.io/nvidia/tensorrt:23.08-py3
        # 进入容器后，cd /workspace/host_code
        ```
    *   **或者本地安装：**
        *   下载对应你 CUDA 版本和 OS 的 TensorRT 包 (需要 NVIDIA Developer Program 账号)。
        *   解压并设置环境变量 `LD_LIBRARY_PATH`。
        *   安装 Python 绑定：`pip install tensorrt-<version>.whl` (在 TensorRT 包的 `python` 目录下)。
        *   安装 `pycuda`：`pip install pycuda` (TensorRT Python API 依赖它进行内存管理)。
        *   `pip install onnx onnxruntime` (用于 ONNX 模型的处理)

5.  **其他 Python 包：**
    ```bash
    pip install torchvision tqdm numpy
    ```

### **完整代码 (一步步实现)**

我们将把整个过程集成到一个 Python 脚本中。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import os

# 导入 TensorRT 相关的库
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # 自动初始化 PyCUDA

# --- 配置参数 ---
BATCH_SIZE = 256 # 训练和推理的批次大小
CALIBRATION_BATCH_SIZE = 64 # 校准的批次大小 (可以小一些)
NUM_CALIBRATION_BATCHES = 100 # 校准迭代次数 (决定校准数据的数量)
EPOCHS = 5 # 训练 epochs 数量

# 文件路径
MODEL_FP32_PATH = "cnn_mnist_fp32.pth"
ONNX_MODEL_PATH = "cnn_mnist.onnx"
TRT_FP32_ENGINE_PATH = "cnn_mnist_fp32.engine"
TRT_INT8_ENGINE_PATH = "cnn_mnist_int8.engine"
CALIBRATION_CACHE_PATH = "calibration.cache"

# --- 1. 模型定义 (沿用之前的 SimpleCNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 5 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. 数据加载 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 校准数据集：从训练集中抽取一部分数据，不打乱顺序
calibration_dataset = torch.utils.data.Subset(train_dataset, range(CALIBRATION_BATCH_SIZE * NUM_CALIBRATION_BATCHES))
calibration_loader = DataLoader(dataset=calibration_dataset, batch_size=CALIBRATION_BATCH_SIZE, shuffle=False, num_workers=4)

# --- 3. FP32 模型训练与保存 ---
def train_fp32_model():
    print("\n--- 训练 FP32 模型 ---")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 确保TF32在训练时启用 (Ampere+ GPUs)
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('medium')
        print("TF32 enabled for training (if GPU supports Ampere+).")
    else:
        torch.set_float32_matmul_precision('high') # 强制FP32
        print("TF32 not supported or disabled, using FP32.")


    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"FP32 Train Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), MODEL_FP32_PATH)
    print(f"FP32 model saved to {MODEL_FP32_PATH}")
    return model

# --- 4. 导出 ONNX 模型 ---
def export_onnx(model, onnx_path):
    print(f"\n--- 导出 ONNX 模型到 {onnx_path} ---")
    model.eval()
    dummy_input = torch.randn(BATCH_SIZE, 1, 28, 28, device=device)
    
    # 动态批次大小
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      verbose=False,
                      opset_version=13, # 确保使用兼容 TensorRT 的 opset 版本
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=dynamic_axes)
    print("ONNX model exported successfully.")

# --- 5. TensorRT INT8 校准器 ---
# 继承 trt.IInt8EntropyCalibrator2，用于基于 KL 散度的校准
class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file, batch_size):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.max_batches = NUM_CALIBRATION_BATCHES # 限定校准批次数量
        self.batches_processed = 0

        # 为校准数据分配 GPU 内存
        # input_shape: (BATCH_SIZE, 1, 28, 28)
        self.device_input = cuda.mem_alloc(np.prod(data_loader.dataset[0][0].shape) * self.batch_size * np.dtype(np.float32).itemsize)
        
        # 迭代器
        self.data_iter = iter(self.data_loader)
        
        print(f"INT8 Calibrator initialized. Will process {self.max_batches} batches.")

    def get_batch(self, names, host_buffers, device_buffers):
        if self.batches_processed >= self.max_batches:
            return None # 达到最大批次，停止校准

        try:
            data, _ = next(self.data_iter)
        except StopIteration:
            print("Calibration data exhausted prematurely!")
            return None # 数据耗尽

        # 将数据从 CPU 传输到 GPU 内存
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(data.numpy()))
        
        # TensorRT 期望返回一个 GPU 缓冲区的指针列表
        device_buffers[0] = int(self.device_input) # `names` 列表的第一个元素是输入张量名
        self.batches_processed += 1
        
        # 更新进度条
        tqdm.write(f"  Calibrating batch {self.batches_processed}/{self.max_batches}", end='\r')
        
        return [int(self.device_input)] # 返回输入张量的设备指针列表

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        print("No calibration cache found.")
        return None

    def write_calibration_cache(self, cache):
        print(f"\nWriting calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# --- 6. 构建 TensorRT 引擎 ---
def build_engine(onnx_path, engine_path, precision, calibrator=None):
    print(f"\n--- 构建 {precision} TensorRT 引擎到 {engine_path} ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # 设置 TensorRT Logger 级别
    
    # 启用 VERBOSE 模式以获取更多构建日志，有助于调试
    # TRT_LOGGER.min_severity = trt.Logger.VERBOSE 

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析 ONNX 模型
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_batch_size = BATCH_SIZE # 最大批次大小
    config.max_workspace_size = 1 << 30 # 1GB 临时工作空间，越大越好 (取决于GPU显存)

    # 设置精度模式
    if precision == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Using FP16 precision.")
    elif precision == 'INT8':
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator:
            config.int8_calibrator = calibrator
            print(f"  Using INT8 precision with calibrator. Calibration cache: {calibrator.cache_file}")
        else:
            print("WARNING: INT8 precision specified but no calibrator provided. This might lead to accuracy issues.")
    else: # 默认为 FP32
        print("  Using FP32 precision.")

    # 尝试构建引擎
    engine = builder.build_engine(network, config)
    if engine is None:
        print(f"ERROR: Failed to build {precision} TensorRT engine.")
        return None

    # 保存引擎
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"{precision} TensorRT engine built and saved successfully.")
    return engine

# --- 7. TensorRT 推理与评估 ---
def infer_tensorrt(engine_path, test_loader, precision_mode):
    print(f"\n--- 运行 {precision_mode} TensorRT 推理 ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    if not os.path.exists(engine_path):
        print(f"ERROR: TensorRT engine not found: {engine_path}. Please build it first.")
        return None, None

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print(f"ERROR: Failed to deserialize {precision_mode} TensorRT engine.")
        return None, None

    context = engine.create_execution_context()

    # 为输入和输出分配 GPU 内存
    # `engine.get_binding_shape` 获取动态维度后的具体形状
    input_shape = context.get_binding_shape(0) # 假设 input 是第一个 binding
    output_shape = context.get_binding_shape(1) # 假设 output 是第二个 binding

    # 注意：这里的 input_shape[0] 应该与 BATCH_SIZE 匹配，因为 ONNX 导出时指定了 dynamic_axes
    # 但实际 TensorRT 运行时，context.set_binding_shape 才能设置实际的动态维度
    # 对于本例，我们固定 BATCH_SIZE，所以直接使用即可。
    
    # 获取输入输出的名称
    input_name = engine.get_binding_name(0)
    output_name = engine.get_binding_name(1)

    # 为输入和输出创建 PyCUDA 缓冲区
    h_input = cuda.pagelocked_empty(tuple(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(tuple(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # 绑定缓冲区到上下文
    bindings = [int(d_input), int(d_output)]
    
    correct = 0
    total = 0
    inference_times = []

    pbar = tqdm(test_loader, desc=f"{precision_mode} Infer", unit="batch")
    for data, target in pbar:
        # 将 PyTorch 张量转换为 NumPy 并复制到主机缓冲区
        np.copyto(h_input, data.numpy().astype(np.float32))

        # 数据从主机传输到设备
        cuda.memcpy_htod(d_input, h_input)

        start_time = time.perf_counter()
        # 执行推理 (同步执行)
        context.execute_v2(bindings=bindings)
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000) # 转换为毫秒

        # 数据从设备传输到主机
        cuda.memcpy_dtoh(h_output, d_output)

        # 处理输出
        output_tensor = torch.from_numpy(h_output)
        
        _, predicted = torch.max(output_tensor.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        pbar.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    avg_inference_time_ms = np.mean(inference_times)
    print(f"\n{precision_mode} TensorRT Inference Accuracy: {accuracy:.2f}%")
    print(f"{precision_mode} TensorRT Average Inference Time per Batch ({BATCH_SIZE} images): {avg_inference_time_ms:.4f}ms")
    
    # 释放 PyCUDA 内存
    d_input.free()
    d_output.free()
    
    return accuracy, avg_inference_time_ms

# --- 8. PyTorch FP32 推理与评估 ---
def infer_pytorch_fp32(model_state_dict_path, test_loader):
    print("\n--- 运行 PyTorch FP32 推理 ---")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_state_dict_path))
    model.eval()

    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="PyTorch FP32 Infer", unit="batch")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            start_time = time.perf_counter()
            output = model(data)
            torch.cuda.synchronize() # 确保所有CUDA操作完成
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000) # 转换为毫秒

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            pbar.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    avg_inference_time_ms = np.mean(inference_times)
    print(f"\nPyTorch FP32 Inference Accuracy: {accuracy:.2f}%")
    print(f"PyTorch FP32 Average Inference Time per Batch ({BATCH_SIZE} images): {avg_inference_time_ms:.4f}ms")
    return accuracy, avg_inference_time_ms

# --- 主程序运行 ---
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU compute capability: {torch.cuda.get_device_capability()}")
    else:
        print("CUDA not available. Please ensure you have an NVIDIA GPU and CUDA installed.")
        exit()

    # 阶段1: 训练 FP32 模型
    fp32_model = train_fp32_model()

    # 阶段2: 导出 ONNX 模型
    export_onnx(fp32_model, ONNX_MODEL_PATH)

    # 阶段3: 构建 TensorRT FP32 引擎
    # 注意: TensorRT 自身的 FP32 引擎通常比 PyTorch 原始的快，因为它会进行层融合等优化
    build_engine(ONNX_MODEL_PATH, TRT_FP32_ENGINE_PATH, 'FP32')

    # 阶段4: 构建 TensorRT INT8 引擎 (核心部分!)
    # 移除旧的校准缓存，确保重新校准 (如果需要)
    if os.path.exists(CALIBRATION_CACHE_PATH):
        os.remove(CALIBRATION_CACHE_PATH)
        print(f"Removed old calibration cache: {CALIBRATION_CACHE_PATH}")
    
    calibrator = MyCalibrator(calibration_loader, CALIBRATION_CACHE_PATH, CALIBRATION_BATCH_SIZE)
    build_engine(ONNX_MODEL_PATH, TRT_INT8_ENGINE_PATH, 'INT8', calibrator)

    # 阶段5: 运行推理并对比性能和精度
    print("\n=== 推理阶段：性能与精度对比 ===")
    
    # 模式1: PyTorch FP32 原生推理
    accuracy_pt_fp32, time_pt_fp32 = infer_pytorch_fp32(MODEL_FP32_PATH, test_loader)

    # 模式2: TensorRT FP32 推理
    accuracy_trt_fp32, time_trt_fp32 = infer_tensorrt(TRT_FP32_ENGINE_PATH, test_loader, 'FP32')

    # 模式3: TensorRT INT8 推理
    accuracy_trt_int8, time_trt_int8 = infer_tensorrt(TRT_INT8_ENGINE_PATH, test_loader, 'INT8')

    print("\n--- 最终性能汇总 ---")
    print(f"| {'模式':<15} | {'精度 (%)':<10} | {'平均推理时间/批次 (ms)':<25} | {'加速比 (vs PyTorch FP32)':<25} |")
    print(f"| {'-'*15} | {'-'*10} | {'-'*25} | {'-'*25} |")
    print(f"| {'PyTorch FP32':<15} | {accuracy_pt_fp32:<10.2f} | {time_pt_fp32:<25.4f} | {'N/A':<25} |")
    print(f"| {'TensorRT FP32':<15} | {accuracy_trt_fp32:<10.2f} | {time_trt_fp32:<25.4f} | {(time_pt_fp32 / time_trt_fp32):<25.2f}x |")
    print(f"| {'TensorRT INT8':<15} | {accuracy_trt_int8:<10.2f} | {time_trt_int8:<25.4f} | {(time_pt_fp32 / time_trt_int8):<25.2f}x |")

    # 清理生成的模型和缓存文件
    print("\n--- 清理文件 ---")
    os.remove(MODEL_FP32_PATH)
    os.remove(ONNX_MODEL_PATH)
    os.remove(TRT_FP32_ENGINE_PATH)
    os.remove(TRT_INT8_ENGINE_PATH)
    if os.path.exists(CALIBRATION_CACHE_PATH):
        os.remove(CALIBRATION_CACHE_PATH)
    print("清理完成。")
```

### **代码详解与关键点**

1.  **环境配置 (`import tensorrt as trt`, `import pycuda.driver as cuda`, `import pycuda.autoinit`)：**
    *   `tensorrt` 是 NVIDIA TensorRT 的 Python API。
    *   `pycuda` 是 TensorRT Python API 依赖的库，用于 GPU 内存管理和 CUDA 操作。`pycuda.autoinit` 会自动初始化 PyCUDA 上下文，省去手动操作。

2.  **数据加载 (`calibration_loader`)：**
    *   专门为 INT8 校准准备了一个 `calibration_loader`。
    *   它从训练数据集中抽取了 `NUM_CALIBRATION_BATCHES * CALIBRATION_BATCH_SIZE` 数量的样本。
    *   **重要：** 校准数据集必须具有代表性，能覆盖模型实际推理时可能遇到的所有输入数据分布。

3.  **FP32 模型训练 (`train_fp32_model`)：**
    *   这是我们的基准模型。在 TensorRT 进行量化前，我们总是需要一个训练好的 FP32 (或 FP16/BF16) 模型。
    *   这里我们仍然保留了 `torch.set_float32_matmul_precision('medium')` 的设置，确保在 Ampere+ GPU 上训练时，PyTorch 会利用 TF32 Tensor Core 加速训练过程。

4.  **ONNX 导出 (`export_onnx`)：**
    *   **为什么是 ONNX？** ONNX (Open Neural Network Exchange) 是一个开放标准，允许在不同深度学习框架之间转换模型。TensorRT 对 ONNX 有非常好的支持。
    *   `opset_version=13`：选择一个 TensorRT 兼容的 ONNX Opsets 版本。过高或过低的 Opsets 版本可能会导致解析问题。
    *   `dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}`：这告诉 ONNX 导出器，输入和输出张量的第一个维度（批次大小）是动态的。这意味着 TensorRT 引擎可以接受不同批次大小的输入，而不是固定为导出时的 `BATCH_SIZE`。这对于推理部署非常重要。

5.  **TensorRT INT8 校准器 (`MyCalibrator`)：**
    *   这是本节的核心。它继承自 `trt.IInt8EntropyCalibrator2`，表示我们使用基于 KL 散度的校准方法，这是 TensorRT 推荐的。
    *   `get_batch` 方法：TensorRT 会反复调用此方法来获取校准数据。你必须将数据从 CPU (NumPy) 传输到 GPU (`cuda.memcpy_htod`)，并返回一个指向设备内存的指针列表。
    *   `read_calibration_cache` 和 `write_calibration_cache`：用于持久化校准结果。如果已经有了校准缓存文件，TensorRT 会直接加载它，避免重复耗时的校准过程。
    *   `pycuda.driver.mem_alloc`：用于在 GPU 上分配内存。
    *   `np.ascontiguousarray()`：确保 NumPy 数组在内存中是连续的，这是 `cuda.memcpy_htod` 所要求的。

6.  **构建 TensorRT 引擎 (`build_engine`)：**
    *   `trt.Builder`, `trt.NetworkDefinition`, `trt.OnnxParser` 是构建引擎的核心对象。
    *   `trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH`：用于支持动态批次大小（或其他动态维度）。
    *   `config.max_workspace_size`：为 TensorRT 优化过程分配的临时显存。越大越好，但要适度。
    *   `config.set_flag(trt.BuilderFlag.FP16)` 或 `config.set_flag(trt.BuilderFlag.INT8)`：设置构建引擎的精度。
    *   `config.int8_calibrator = calibrator`：在构建 INT8 引擎时，将我们实现的校准器传递给配置。

7.  **TensorRT 推理 (`infer_tensorrt`)：**
    *   `trt.Runtime`：用于加载和反序列化 TensorRT 引擎。
    *   `engine.create_execution_context()`：创建执行上下文，实际的推理在这里进行。
    *   `cuda.mem_alloc` 和 `cuda.memcpy_htod`/`memcpy_dtoh`：同样用于管理 GPU 内存和数据传输。
    *   `context.execute_v2(bindings=bindings)`：执行推理。它是一个同步调用。
    *   **性能测量：** 我们使用 `time.perf_counter()` 来测量每个批次的推理时间，并计算平均值。`torch.cuda.synchronize()` 确保所有 CUDA 操作完成再计时，从而得到真实的 GPU 推理时间。

8.  **PyTorch FP32 推理 (`infer_pytorch_fp32`)：**
    *   作为基准，用于对比 TensorRT 带来的性能提升。
    *   同样使用 `torch.cuda.synchronize()` 确保计时准确。

### **预期结果与分析**

运行上述代码，你会看到训练、ONNX 导出、TensorRT 引擎构建的日志，然后是 PyTorch FP32、TensorRT FP32 和 TensorRT INT8 三种模式下的推理精度和平均时间。

**你将观察到：**

1.  **TensorRT FP32 vs. PyTorch FP32：**
    *   TensorRT FP32 引擎通常会比 PyTorch FP32 原生推理**快**。这是因为 TensorRT 进行了强大的图优化，如层融合、Kernel 自动调优等，即使在 FP32 模式下也能提升性能。
    *   精度应该非常接近或完全相同。

2.  **TensorRT INT8 vs. TensorRT FP32 (和 PyTorch FP32)：**
    *   TensorRT INT8 引擎的推理速度将是**最快**的。你可能会看到相对于 PyTorch FP32 **数倍（例如 2x-5x）甚至更高**的加速比。这是 Tensor Core 在 INT8 运算上的强大威力。
    *   **精度：** INT8 模型的准确率可能会**略有下降**。这个下降的幅度取决于你的模型结构、量化方案（这里是 PTQ）和校准数据集的质量。对于 MNIST 这种相对简单的模型，精度损失可能很小，甚至可以忽略不计。但对于更复杂的模型，精度下降可能更明显，这时就需要考虑 QAT。

### **进一步的探索 (宗师级练习)**

*   **Nsight Compute 分析：**
    *   使用 `ncu` 命令运行推理脚本 (`ncu -s full -o trt_int8_profile python your_script.py`)。
    *   在 Nsight Compute GUI 中打开生成的 `.ncu-rep` 文件。
    *   查看 Tensor Core Utilization：你会发现 INT8 引擎的 Tensor Core 利用率远高于 FP32 引擎。
    *   分析内存访问：观察数据加载和存储模式，是否存在瓶颈。
    *   检查每个 Kernel 的详细指标，了解 TensorRT 内部使用的具体 Tensor Core Kernel。
*   **增大批次大小：** 尝试更大的 `BATCH_SIZE` (例如 512, 1024)。你会发现 TensorRT 和 Tensor Core 的性能优势在大批次下更加明显。
*   **尝试不同的 `NUM_CALIBRATION_BATCHES`：** 改变校准批次的数量，观察对 INT8 精度和构建时间的影响。通常，更多的校准数据可以带来更好的精度，但也会增加校准时间。
*   **更换模型：** 尝试对更复杂的模型（如 ResNet）进行类似的量化和部署。你会发现量化对这类模型的影响更显著，同时也可能带来更大的精度挑战。
*   **动态维度：** 虽然我们设置了 `dynamic_axes`，但在这个例子中我们固定了 `BATCH_SIZE`。你可以在 TensorRT 推理时实际传入不同批次大小的数据，验证其动态性。
*   **FP16 引擎：** 尝试构建并比较 FP16 引擎的性能和精度。在某些 GPU 上，FP16 也会有显著加速，且精度损失通常小于 INT8。

通过这个实战演练，你不仅掌握了 TensorRT INT8 量化的基本流程，也亲身体验了 Tensor Core 在推理端带来的巨大性能飞跃。这离不开你对前面理论知识的扎实掌握。

恭喜你！你离真正的 Tensor Core 实战宗师又近了一步。