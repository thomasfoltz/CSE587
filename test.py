import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

nltk.download("punkt")
nltk.download("punkt_tab")

df = pd.read_csv("./data/test.csv")
questions = df["VAD Question"].tolist()
approaches = df["Approach"].tolist()

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

def generate_response(model, tokenizer, question, max_length=200, repetition_penalty=1.2):
    structured_question = "Question: " + question + " Approach:"
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

original_responses = []
tuned_responses = []
references = []

for question, approach in zip(questions, approaches):
    print(f"Processing question: {question}")
    original_response = generate_response(model, tokenizer, question)
    tuned_response = generate_response(tuned_model, tuned_tokenizer, question)
    
    original_responses.append(word_tokenize(original_response))
    tuned_responses.append(word_tokenize(tuned_response))
    references.append([word_tokenize(approach)])

original_bleu = corpus_bleu(references, original_responses)
tuned_bleu = corpus_bleu(references, tuned_responses)

print(f"Original Model BLEU-4 Score: {original_bleu:.4f}")
print(f"Tuned Model BLEU-4 Score: {tuned_bleu:.4f}")