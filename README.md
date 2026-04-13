# CIFAR-10 深度学习架构对比：ResNeXt50 vs DenseNet121

## 📌 项目简介
本项目探究并对比了两种经典卷积神经网络架构（ResNeXt50 与 DenseNet121）在 CIFAR-10 图像分类任务上的性能表现。实验重点对比了**从头训练 (Training from Scratch)** 与**迁移学习/微调 (Fine-tuning)** 在收敛速度和最终准确率上的巨大差异。

本实验作为浙江大学（ZJU）深度学习相关课程的 Project 提交。

## 🛠️ 环境依赖
本项目在 Ubuntu 22.04 (WSL2) 环境下开发，并使用 NVIDIA RTX 4080 (Laptop) 进行了硬件加速验证。
* Python 3.11
* PyTorch 2.5.1
* Torchvision 0.20.1
* CUDA 12.1
* Matplotlib & NumPy (用于数据可视化)

## 📁 核心文件结构
* `Project-2.py`: 核心训练与验证脚本，包含了模型定义、数据加载 (CIFAR-10, Resize to 224)、训练循环与准确率记录。
* `make_plots.py`: 数据可视化脚本 1，用于生成四组实验的 Epoch-Accuracy 全景对比折线图。
* `plot_finetune.py`: 数据可视化脚本 2，用于生成 Finetune 模式下 ResNeXt 和 DenseNet 的高精度细节对比图。
* `Deep_Learning_Report_ZJU.tex`: 采用 LaTeX 编写的标准实验报告源码，包含详细的图表和结果分析。

## 🚀 实验结果摘要
在统一的超参数设置下（SGD, LR=0.001, Epochs=10, Batch Size=32），实验结果如下：

| 模型架构 | 训练模式 | 最佳验证准确率 | 总训练时间 |
| :--- | :--- | :--- | :--- |
| ResNeXt50 | Scratch | 84.61% | ~66 min |
| **ResNeXt50** | **Finetune** | **97.13%** | **~66 min** |
| DenseNet121 | Scratch | 87.05% | ~52 min |
| DenseNet121 | Finetune | 97.05% | ~52 min |

**结论要点：**
1. **迁移学习优势极强**：Finetune 模式在第一轮即达到 94%+ 准确率，最终收敛至 ~97%，远超从头训练的 85% 左右。
2. **架构特性博弈**：ResNeXt50 凭借分组卷积在预训练加持下获得了略高的极限精度 (97.13%)；而 DenseNet121 则展现了更高的参数效率和训练速度。

## ⚙️ 运行指南
1. 确保已安装所需的 Conda 虚拟环境及依赖。
2. 运行训练脚本：
   ```bash
   python Project-2.py
