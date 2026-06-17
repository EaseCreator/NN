Project-5: 基于 LoRA 的 Qwen2.5-1.5B 医疗大模型微调

本项目使用 Unsloth 框架对 Qwen2.5-1.5B 大语言模型进行医疗垂直领域的指令微调（SFT），使其具备专业的中医问诊与处方推荐能力。

1. 环境与依赖 (Dependencies)

本项目需要支持 CUDA 的 GPU 环境（建议显存 8GB 及以上，如 RTX 4070）。建议在 Conda 虚拟环境中安装以下核心依赖：

# 1. 安装 PyTorch (请根据你的 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 2. 安装 Unsloth 及相关加速库 (核心训练框架)
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# 3. 安装其他常规工具库
pip install transformers datasets


2. 数据集 (Dataset)

本项目使用的是开源中文医疗对话数据集 shibing624/medical 中的 finetune 子集。

为了避免代码自动下载时因网络波动或框架限制导致卡死，请手动下载该数据集文件，并将其放置在项目根目录（与 Python 脚本同级）。

目标文件名: train_zh_0.json (大小约 1.34GB)

Hugging Face 官方下载链接: 点击此处下载

国内镜像加速下载链接: 点击此处下载

3. 模型权重说明 (Model Weights)

本项目涉及两部分模型权重：基座模型权重与微调后的 LoRA 权重。

基座模型 (Base Model)

代码默认加载 Qwen2.5-1.5B 模型。如果你已经将模型下载到本地（如本次实验中的 ./Qwen-1.5B-Local），请在代码的 model_name 参数中指向该本地路径。
如果没有下载，也可以直接通过 Hugging Face 或魔搭社区 (ModelScope) 获取：

Hugging Face 路径: Qwen/Qwen2.5-1.5B

魔搭社区 (国内) 路径: qwen/Qwen2.5-1.5B

微调权重 (LoRA Weights)

如何获取: 运行 train.py 脚本（耗时约15分钟）。训练结束后，代码会自动将微调好的 LoRA 权重持久化保存至当前目录下的 ./lora_medical/ 文件夹中。

如何使用: 运行 infer.py 脚本时，代码会自动从 ./lora_medical/ 目录读取这些权重并与基座模型合并，直接进行推理测试，无需再次联网下载或重新训练。