import re

def extract_answer(text):
    match = re.search(r'\$\\boxed\{(.*?)\}\$', text)
    if match:
        return match.group(1)
    else:
        return text ## If the answer is not in the format of $\\boxed{}$, return the original text
    
def extract_s1_answer(text):
    match = re.search(r'(.+?)}\$', text)
    return match.group(1) if match else text

def extract_qwen_answer(text):
    if text is None:
        return None
    boxed_tag = r"\boxed{"
    tag_len = len(boxed_tag)
    start_pos = 0
    start = text.find(boxed_tag, start_pos)
    if start == -1:
        return None
    content_start = start + tag_len
    counter = 1
    content_end = content_start
    while content_end < len(text) and counter > 0:
        if text[content_end] == '{':
            counter += 1
        elif text[content_end] == '}':
            counter -= 1
        content_end += 1
    if counter == 0:
        return text[content_start:content_end-1]
    else:
        return None
    
def judge_answer(question, generation, ground_truth, client):
    if generation == ground_truth:
        return True
    else:
        prompt = f"Given the question: {question}\nPlease help me determine whether the following two answers to the quesiton are mathematical equivalent, ignoring any errors that might be caused by the format and ignoring any numerical units\' difference.\nThe first answer: {generation}\nThe second answer: {ground_truth}\nPlease give your answer in the following format: 'Equivalent' or 'Not Equivalent'."
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
        elif 'Not Equivalent' in decision:
            return False
        elif 'Equivalent' in decision:
            return True
        else:
            return False
        
def remove_duplicate_sentences(text, n=5):
    pattern = r'\.\s+'
    matches = list(re.finditer(pattern, text))
    sentences = []
    separators = []
    prev_end = 0
    for match in matches:
        sentence = text[prev_end:match.start()]
        separator = match.group()
        sentences.append(sentence)
        separators.append(separator)
        prev_end = match.end()
    if prev_end < len(text):
        sentences.append(text[prev_end:])
    extended_separators = separators + ['']
    seen = set()
    cut_index = None
    for i in range(len(sentences) - n + 1):
        group_sentences = sentences[i:i+n]
        group = ". ".join([s.rstrip('.') for s in group_sentences]) + ". "
        if group in seen:
            cut_index = i
            break
        seen.add(group)
    if cut_index is not None:
        k = cut_index
    else:
        k = len(sentences)
    reconstruct = ''.join([sentences[i] + extended_separators[i] for i in range(k)])
    return reconstruct