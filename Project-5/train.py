import os
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048 

print("1. 加载基座模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./Qwen-1.5B-Local", # 读取下好的本地模型
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

print("2. 准备数据集...")
# 截取 10000 条，大约 15 分钟即可完成微调
dataset = load_dataset("json", data_files="train_zh_0.json", split="train[:10000]")
dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,         
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine", 
        seed = 3407,
        output_dir = "outputs",
        save_strategy = "no",
    ),
)

print("3. 开始微调训练...")
trainer.train()

print("4. 训练完成，正在将 LoRA 权重永久保存到本地...")
model.save_pretrained("lora_medical")
tokenizer.save_pretrained("lora_medical")
print("保存成功！可以放心关闭程序了。")