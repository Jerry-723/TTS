import re

def extract_answer(text):
    match = re.search(r'\\boxed\{(.+?)\}', text)
    if match:
        return match.group(1)
    else:
        return None
    
def judge_answer(generation, ground_truth, client):
    if generation == ground_truth:
        return True
    else:
        prompt = f"Please help me determine whether the following two answers are equal, ignoring any errors that might be caused by the format and ignoring any numerical units\' difference.\nThe first answer: {generation}\nThe second answer: {ground_truth}\nPlease give your answer in the following format: 'Equal' or 'Not Equal'."
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
        if decision == 'Equal':
            return True
        elif decision == 'Not Equal':
            return False
        else:
            return False ## GPT4o doesn't follow the instruct, giving a False by default