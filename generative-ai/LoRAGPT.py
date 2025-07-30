from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token

def tokenize_fn(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

model = AutoModelForCausalLM.from_pretrained("gpt")

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 uses 'c_attn' for QKV projections
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora-gpt2",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    learning_rate=1e-4,
    remove_unused_columns=False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(1000)),  # small subset for demo
    eval_dataset=tokenized["validation"].select(range(200)),
    data_collator=data_collator
)

trainer.train()
# Save only LoRA adapter
model.save_pretrained("lora-gpt2-adapter")