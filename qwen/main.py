from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 【正确写法：Windows 相对路径】
model_path = "./Qwen1___5-0___5B-Chat"  # 复制这个！

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# 推理
prompt = "你好"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("\n✅ Qwen1.5 成功运行！")
print(response)