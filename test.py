import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

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

generated_responses = "./data/generated_responses.csv"

if os.path.exists(generated_responses):
    df = pd.read_csv(generated_responses)
    questions, original_responses, tuned_responses = df["Question"].tolist(), df["Original_Response"].tolist(), df["Tuned_Response"].tolist()

else:
    print("Generating responses...")
    original_responses = [generate_response(model, tokenizer, q) for q in questions]
    tuned_responses = [generate_response(tuned_model, tuned_tokenizer, q) for q in questions]
    
    data = {
        "Question": questions,
        "Original_Response": original_responses,
        "Tuned_Response": tuned_responses
    }
    df = pd.DataFrame(data)
    df.to_csv(generated_responses, index=False)

encoder = SentenceTransformer('all-mpnet-base-v2')
def calculate_semantic_scores(questions, generated_responses, ground_truths):
    question_embs = encoder.encode(questions)
    response_embs = encoder.encode(generated_responses)
    truth_embs = encoder.encode(ground_truths)
    
    qa_similarity = cosine_similarity(response_embs, truth_embs).diagonal()
    context_similarity = cosine_similarity(response_embs, question_embs).diagonal()
    
    return {
        'mean_semantic_similarity': qa_similarity.mean(),
        'mean_context_relevance': context_similarity.mean()
    }

original_scores = calculate_semantic_scores(questions, original_responses, approaches)
tuned_scores = calculate_semantic_scores(questions, tuned_responses, approaches)

print(original_scores)
print(tuned_scores)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def calculate_rouge(generated, references):
    scores = [scorer.score(ref, gen)['rougeL'].fmeasure 
             for gen, ref in zip(generated, references)]
    return sum(scores)/len(scores)

print(f"Original ROUGE-L: {calculate_rouge(original_responses, approaches):.4f}")
print(f"Tuned ROUGE-L: {calculate_rouge(tuned_responses, approaches):.4f}")