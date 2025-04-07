import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import bitsandbytes as bnb
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto")
# model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",  # Use float16 for computation
    bnb_4bit_use_double_quant=True,   # Enable double quantization
    bnb_4bit_quant_type="nf4"         # Use NormalFloat4 (NF4) quantization
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    device_map="auto",
    quantization_config=bnb_config, # Use the updated configuration
    use_cache=False,
    torch_dtype=torch.float16,
)
model.gradient_checkpointing_enable()

data_files = {"train": "train.json", "validation": "val.json"}
dataset = load_dataset("json", data_files=data_files)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["explanation"], truncation=True, padding="max_length", max_length=tokenizer.model_max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "explanation"])

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False,
    gradient_accumulation_steps=2,  # Accumulate gradients
    fp16=True,  # Enable mixed precision
    optim="paged_adamw_8bit",  # Use 8-bit optimizer
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()
eval_results = trainer.evaluate()
print(eval_results)

model_path = "./fine_tuned_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# tuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
# tuned_model = AutoModelForCausalLM.from_pretrained(model_path)

# if tuned_tokenizer.pad_token is None:
#     tuned_tokenizer.pad_token = tuned_tokenizer.eos_token

# sample_input = "Describe defect: Missing Hole, (371.88, 147.83), 71.05%"

# input_ids = tuned_tokenizer.encode(sample_input, return_tensors="pt")
# output_ids = tuned_model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

# generated_text = tuned_tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(generated_text)