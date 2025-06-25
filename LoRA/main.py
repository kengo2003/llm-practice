from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel

base_model = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# チューニング前
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
pipe_base = pipeline("text2text-generation",model=base_model,tokenizer=tokenizer)

# チューニング後
lora_model = PeftModel.from_pretrained(base_model, "lora-flan/checkpoint-1800")
pipe_lora = pipeline("text2text-generation",model=lora_model,tokenizer=tokenizer)

prompt = "What's the capital of japan?"
generate_args = {
    "max_new_tokens": 20,
    "do_sample": True,
    "temperature": 0.7,
    "num_beams": 1
}

# 実行
result_base = pipe_base(prompt, **generate_args)[0]["generated_text"]
result_lora = pipe_lora(prompt, **generate_args)[0]["generated_text"]

print("Prompt:", prompt)
print("Generate args:", generate_args)
print("チューニング前:", result_base)
print("チューニング後:", result_lora)
print("Base model result:", pipe_base(prompt, **generate_args))
print("LoRA model result:", pipe_lora(prompt, **generate_args))

