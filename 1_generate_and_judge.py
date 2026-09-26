import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# --- Configuration ---
NUM_SAMPLES = 1500
SLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS_OUTPUT_FILE = "nlp_execution_labels.npy"

# --- API Setup ---
# reads OPENAI_API_KEY from the environment
client = OpenAI()

print("Loading SLM...")
tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(SLM_MODEL_ID, device_map="auto")

print("Loading MMLU Dataset...")
# We load the "all" config and use the test split which has ~14k diverse questions
dataset = load_dataset("cais/mmlu", "all", split="test")
subset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

labels = []

def oracle_judge(question, choices_text, true_answer_letter, slm_answer):
    """
    Oracle evaluates if the SLM picked the correct multiple choice letter.
    """
    evaluation_prompt = f"""
    You are a strict grader. 
    
    Question: {question}
    Choices: 
    {choices_text}
    
    The Correct Answer is: {true_answer_letter}
    
    The Student (SLM) generated this text: "{slm_answer}"
    
    Did the student select the correct answer ({true_answer_letter})? 
    Output strictly the number 1 for Yes, or 0 for No.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.0, 
            max_tokens=5
        )
        score_text = response.choices[0].message.content.strip()
        return 1 if "1" in score_text else 0
    except Exception as e:
        return 0 

print("Generating and Judging...")
for i, item in enumerate(subset):
    question_text = item['question']
    choices = item['choices']
    
    # MMLU gives answers as 0, 1, 2, 3. We convert to A, B, C, D.
    letters = ['A', 'B', 'C', 'D']
    true_answer_letter = letters[item['answer']]
    
    choices_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    prompt_text = f"Question: {question_text}\n{choices_text}\nAnswer:"
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
    input_length = inputs.input_ids.shape[1]
    slm_answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    score = oracle_judge(question_text, choices_text, true_answer_letter, slm_answer)
    labels.append(score)
    
    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{NUM_SAMPLES} | Current Success Rate: {(sum(labels)/(i+1))*100:.1f}%")

np.save(LABELS_OUTPUT_FILE, np.array(labels))
print(f"Total Successes: {sum(labels)}/{NUM_SAMPLES}")