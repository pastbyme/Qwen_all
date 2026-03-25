from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 你本地真实正确的路径（已经修好）
model_path = r"D:\Qwen\qwen\Qwen2___5-0___5B-Instruct"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    local_files_only=True
)

# ✅ 这里修复了！！！
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# 推理
prompt = "你好"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("✅ Qwen2.5 运行成功！")
print(response)