import re
import json
import os

def extract_answer(text):
    match = re.search(r'\$\\boxed\{(.*?)\}\$', text)
    if match:
        return match.group(1)
    else:
        return text ## If the answer is not in the format of $\\boxed{}$, return the original text
    
def judge_answer(question, generation, ground_truth, client):
    if generation == ground_truth:
        return True
    else:
        prompt = f"Given the question: {question}\nPlease help me determine whether the following two answers to the quesiton are equivalent, ignoring any errors that might be caused by the format and ignoring any numerical units\' difference.\nThe first answer: {generation}\nThe second answer: {ground_truth}\nPlease give your answer in the following format: 'Equivalent' or 'Not Equivalent'."
        response = client.chat.completions.create(
            model = "gpt-4o",
            n = 1,
            messages = [
                {
                    'role': "user",
                    'content': prompt
                }
            ],
            temperature = 0.0
        )
        decision = response.choices[0].message.content
        if decision == 'Equivalent':
            return True
        elif decision == 'Not Equivalent':
            return False
        else:
            return False ## GPT4o doesn't follow the instruct, giving a False by default
        
        
def save_to_json(data, data_path):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')