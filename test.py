import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    use_cache=False,
)

tuned_model_path = "./researcher-llama-3.2-3B"
tuned_tokenizer = AutoTokenizer.from_pretrained(tuned_model_path)
tuned_model = AutoModelForCausalLM.from_pretrained(tuned_model_path)

question = "How to detect anomalies?"

def generate_response(model, tokenizer, question, max_length=200, repetition_penalty=1.2):
    structured_question = "Question: " + question + " Approach:"
    # inputs = tokenizer(structured_question, return_tensors="pt", padding=True, truncation=True)
    inputs = tokenizer(structured_question, return_tensors="pt", truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

fine_tuned_response = generate_response(tuned_model, tuned_tokenizer, question)
print("\nFine-tuned llm response:", fine_tuned_response)

original_response = generate_response(model, tokenizer, question)
print("\nOriginal llm response:", original_response)
