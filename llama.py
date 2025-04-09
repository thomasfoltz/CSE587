import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import bitsandbytes as bnb
import json
from sklearn.metrics import classification_report, accuracy_score
import random

model_path = "./AGNews-Llama-3.2-3B"
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", device_map="auto")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-3B",
#     device_map="auto",
#     quantization_config=bnb_config,
#     use_cache=False,
# )

# data_files = {"train": "train.json", "validation": "val.json"}
# dataset = load_dataset("json", data_files=data_files)

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# def tokenize_function(examples):
#     return tokenizer(
#         examples["text"],
#         text_target=examples["label"],
#         padding=True,
#         truncation=True,
#     )

# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,
# )

# training_args = TrainingArguments(
#     output_dir="./results",
#     optim="paged_adamw_8bit",
#     eval_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     save_strategy="epoch",
#     logging_dir="./logs",
#     logging_steps=10,
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     push_to_hub=False,
#     gradient_accumulation_steps=32,
#     fp16=True,
# )

# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj, gate_proj", "down_proj", "up_proj"]
# )

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
# )

# trainer.train()
# eval_results = trainer.evaluate()
# print(eval_results)

# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)

tuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
tuned_model = AutoModelForCausalLM.from_pretrained(model_path)


def classify_news(text):
    # prompt = f"""Classify this news article into one category: world, sports, business, sci/tech.
    # Article: {text}
    # Category:"""

    prompt = f"""Classify this news article into one category: sports, business, sci/tech.
    Article: {text}
    Category:"""
    
    inputs = tuned_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    output_ids = tuned_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=3,
        num_beams=4,
        early_stopping=True,
        pad_token_id=tuned_tokenizer.eos_token_id
    )
    
    prediction = tuned_tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    ).strip().lower()
    
    # Map to original labels
    return prediction.split()[0]  # Take first token

# Load the validation dataset
with open("val.json", "r") as f:
    validation_data = json.load(f)

true_labels = []
predicted_labels = []
random.seed(42)
sampled_entries = random.sample(validation_data, 50)

for entry in sampled_entries:
    text = entry["text"]
    true_label = entry["label"]
    if true_label == "world":
        continue
    predicted_label = classify_news(text)
    if predicted_label not in ["sports", "business", "sci/tech"]:
        continue
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    print(f"True: {true_label}, Predicted: {predicted_label}")

# Calculate evaluation metrics
print("Classification Report:")
# print(classification_report(true_labels, predicted_labels, labels=["world", "sports", "business", "sci/tech"]))
print(classification_report(true_labels, predicted_labels, labels=["sports", "business", "sci/tech"]))

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")