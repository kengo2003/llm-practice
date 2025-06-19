from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "What is strategic decision making?"
inputs = tokenizer(prompt, return_tensors="pt")

# 推論
outputs = model.generate(**inputs, max_new_tokens=50)

# 結果をデコード
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)