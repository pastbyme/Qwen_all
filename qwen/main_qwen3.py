from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型路径
model_path = r"D:\Qwen\qwen\qwen\Qwen3-4B"

# GPU 加载模型（快 10~20 倍）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
)

# 推理
prompt = "你好"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("✅ Qwen3-4B GPU 运行成功！")
print(response)