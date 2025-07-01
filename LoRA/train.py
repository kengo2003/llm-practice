from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint

model_name = "google/flan-t5-small"
dataset = load_dataset("json", data_files="../train_data/train.jsonl", split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# LoRA設定
peft_config = LoraConfig(
  r=16,
  lora_alpha=32,
  target_modules=["q","v","k","o"],
  lora_dropout=0.1,
  bias="none",
  task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model,peft_config)

# データ前処理
def preprocess(example):
  # 入力テキストの作成
  if example['input']:
    prompt = f"{example['instruction']}\n{example['input']}"
  else:
    prompt = example['instruction']
  
  # 入力のトークナイゼーション
  inputs = tokenizer(
    prompt,
    truncation=True,
    padding="max_length",
    max_length=64,
    return_tensors=None
  )
  
  # 出力のトークナイゼーション
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(
      example["output"],
      truncation=True,
      padding="max_length",
      max_length=24,
      return_tensors=None
    )
  
  # 必要なフィールドのみを返す
  return {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "labels": labels["input_ids"]
  }

# データセットの前処理
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# 学習設定
training_args = TrainingArguments(
  output_dir="./lora-flan",
  per_device_train_batch_size=1,
  num_train_epochs=10,
  logging_steps=1,
  save_strategy="epoch",
  # fp16=True,  #MPSでは使用不可
  save_total_limit=2,
  # MPS用の最適化設定
  dataloader_pin_memory=False,  # MPSでは不要
  remove_unused_columns=False,  # データセットの列を保持
  # 学習率調整
  learning_rate=1e-4,
  warmup_steps=20,
  # 正則化
  weight_decay=0.05,  # 重み減衰
)

# データコレーター設定
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
)

# カスタムトレーナークラス（PEFTモデル用）
class PeftSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        # generation_configエラーを回避
        if 'args' in kwargs and not hasattr(kwargs['args'], 'generation_config'):
            kwargs['args'].generation_config = None
        super().__init__(*args, **kwargs)
        # PEFTモデルの警告を回避
        if hasattr(self.model, 'label_names'):
            self.label_names = self.model.label_names
        else:
            self.label_names = []

trainer = PeftSeq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset,
  data_collator=data_collator,
  tokenizer=tokenizer,  # tokenizerを明示的に渡す
)

trainer.train()