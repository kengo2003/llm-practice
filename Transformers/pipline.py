from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

output = generator("What is strategic decision making?", max_new_tokens=50)
print(output[0]['generated_text'])