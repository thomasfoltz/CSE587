import os
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def preprocess_csv(file_path, output_path):
    df = pd.read_csv(file_path)
    df["input"] = "Question: " + df["VAD Question"]
    df["target"] = "Approach: " + df["Approach"]
    df[["input", "target"]].to_csv(output_path, index=False)

def tokenize_function(examples):
    inputs = tokenizer(
        examples["input"],
        text_target=examples["target"],
        padding=True,
        truncation=True,
        max_length=512,
    )
    return inputs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", device_map="auto")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    quantization_config=bnb_config,
    use_cache=False,
)

preprocess_csv("./data/train.csv", "./data/train_preprocessed.csv")
preprocess_csv("./data/val.csv", "./data/val_preprocessed.csv")

data_files = {"train": "./data/train_preprocessed.csv", "validation": "./data/val_preprocessed.csv"}
dataset = load_dataset("csv", data_files=data_files)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["input", "target"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results",
    optim="paged_adamw_8bit",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=50,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False,
    gradient_accumulation_steps=32,
    bf16=True,
    logging_steps=10
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.10,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj, gate_proj", "down_proj", "up_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()
eval_results = trainer.evaluate()

model_path = "./researcher-llama-3.2-3B"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)