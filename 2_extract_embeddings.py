import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
NUM_SAMPLES = 1500  # Must match File 1
SLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDINGS_OUTPUT_FILE = "nlp_slm_embeddings.npy"

print("Loading SLM...")
tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(SLM_MODEL_ID, device_map="auto")

print("Loading MMLU Dataset (Same Subset)...")
dataset = load_dataset("cais/mmlu", "all", split="test")
subset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

embeddings = []

print("Extracting Hidden States...")
for i, item in enumerate(subset):
    question_text = item['question']
    choices = item['choices']
    choices_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    prompt_text = f"Question: {question_text}\n{choices_text}\nAnswer:"
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_state = outputs.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
    embeddings.append(hidden_state)
    
    if (i + 1) % 100 == 0 or i == NUM_SAMPLES - 1:
        print(f"Extracted {i+1}/{NUM_SAMPLES}")

np.save(EMBEDDINGS_OUTPUT_FILE, np.array(embeddings))
print(f"Saved {NUM_SAMPLES} embeddings.")