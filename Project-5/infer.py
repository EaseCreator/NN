from unsloth import FastLanguageModel
from transformers import TextStreamer

print("1. 正在加载保存的微调模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_medical", 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

FastLanguageModel.for_inference(model)
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

test_cases = [
    "我最近总是失眠多梦，白天没精神，舌苔发白，感觉身体很虚。",
    "大夫，我最近总是干咳，没有痰，喉咙很干，而且下午手脚心发热，有时候还盗汗。",
    "我吃饭总是没胃口，吃一点就觉得肚子胀，大便也不成形，人总是觉得很疲乏。"
]

print("\n================ 模型推理展示 ================")
for case in test_cases:
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "请作为一名专业的老中医，为患者进行诊断并给出调理建议。", 
            case, 
            "", 
        )
    ], return_tensors = "pt").to("cuda")
    
    print(f"\n【患者主诉】: {case}")
    print("【中医诊断】:")
    
    _ = model.generate(
        **inputs, 
        streamer = text_streamer, 
        max_new_tokens = 256,        
        temperature = 0.3,           
        repetition_penalty = 1.2,    
    )
    print("\n" + "=" * 50)