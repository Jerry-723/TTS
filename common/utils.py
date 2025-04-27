import re
from openai import OpenAI
import random

api_key_list = ['sk-KcTMOKcMum8zgDlbyQQSH3HBexYbK0UP6y4RcVM0nic8W0dg',
'sk-G4hrgPh89BOnhhYOtez3uStZslOHROGwrNtZupRlLN40KTz3',
'sk-lQMrAPk63dWr3ilk7WlaenUTUAXnbhIoNCX7IeDd0D213d2v',
'sk-RHAN4qQVQ3e6yLfpJybkoSUQXdKz73USedwypJKAHwdBxLiN',
'sk-HJjPDMUo1AiSWE8V7Skxs2y2jqOspvvIi6qI9c9f8Ts17KE6',
'sk-DyusJQhpsr2hQX7wga18wNrtuJqJcJ6Q3TpggRnUZCRRsxZK',
'sk-IKRbR62e3kdD8cvzWQGBtRLtTEoQnW8WfgUsrYaZBtMfnO7x',
'sk-tCfTxvsnqgisCNMCq8uANIcBsXRf8UEh4IJxycIk0m9aDis9',
'sk-bPLi6N07l4VhlafirkEamvM08CKLjG1EWALNHHq5h9Qv84Bd',
'sk-527iPvnopmtyAmUapO237j5KjHEjNF0ci6EOCfVYYpN2kIwV',
'sk-eukApWhh6phJXXmIWMkKSCUTYdjzq1bKHTAf4mVnvWpKSRhb',
'sk-HDEWmw0420jcZjm33PPdYwNGKHIVhbFluUUSM1AIkrQWoUmB',
'sk-RzDWA95KpwoJ8y9ewhJkO9UkrJ2bFn6y9YrEJ67ON9gH2rnx',
'sk-n2hvLtD5Q2pgGou8lVmAmAf6M4EmTZ10Ly3j2YfMxLAta34y',
'sk-3WVq2A8dCshz93d8WNLzy3MvUhbs568hsjwdfexvd9QU38Ju',
'sk-fF76WeTmAPu5aCfUCJBGoZISV0FBTaDgjd6kA9lLpNztkeL4',
'sk-qxOi75HseirhDuNXSgSL8joqKzQYEiFK3LuVINCCXwJfqUgy',
'sk-XO47DC9CrRACPtpFqULReOUWtgZyd54daguQCghy2cQRWWbf',
'sk-tI9So0Noqf6XfiqMVwEZlmncONwhDdInqZr9BeWdqyeL7g7A',
'sk-wM7vmRzqYw1EjJLyJElhTKlGpeZdowUvdjt3w9a6C3mdEr5g',
'sk-9hnrWYYkwmvNrugeS17guaOGFAQZRV8VGZudjrmphhiX6KEs',
'sk-QZStF6GiUwkkZzYOz8dUNTBl9nC43NKW3cBin3Hp8qGpAZ7x',
'sk-Dz3WLuPJNg1RLDWOVW5Lb3MvFO1k5wdVKt6aETK4LY8pbUKT',
'sk-me5Lw8MqwgapzRsSh3kKbNRPGQeSRVfzSfAlMi0yjEzA8Tc8',
'sk-fCowlik54cOpdpvykNlCB48CIciAwYzIqoaMZ9kV4pmiB2zB',
'sk-mUMfjkEomXwxJal5XgpXNjVJciDX2GqSOIlgo6ilF7S6neXk',
'sk-hyF62L98CqXl7whBVNHFloHWx7rOecyQVFVHuCfllXYvfngh',
'sk-NTQchhYmLxE4iVWid86qOLdDQOBfKst6g8wU9LQOZPZucVtV']

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
    
# def judge_answer(question, generation, ground_truth, client):
#     if generation == ground_truth:
#         return True
#     else:
#         prompt = f"Given the question: {question}\nPlease help me determine whether the following two answers to the quesiton are mathematical equivalent, ignoring any errors that might be caused by the format and ignoring any numerical units\' difference.\nThe first answer: {generation}\nThe second answer: {ground_truth}\nPlease give your answer in the following format: 'Equivalent' or 'Not Equivalent'."
#         response = client.chat.completions.create(
#             model = "gpt-4o",
#             n = 1,
#             messages = [
#                 {
#                     'role': "user",
#                     'content': prompt
#                 }
#             ],
#             temperature = 0.0
#         )
#         decision = response.choices[0].message.content
#         if decision == 'Equivalent':
#             return True
#         elif decision == 'Not Equivalent':
#             return False
#         elif 'Not Equivalent' in decision:
#             return False
#         elif 'Equivalent' in decision:
#             return True
#         else:
#             return False

def judge_answer(question, generation, ground_truth):
    if generation == ground_truth:
        return True
    elif is_equiv(generation, ground_truth):
        return True
    else:
        api_index = random.randint(0, len(api_key_list) - 1)
        client = OpenAI(
            base_url="https://chataiapi.com/v1",
            api_key=api_key_list[api_index]
        )
        prompt = f"Given the question: {question}\nPlease help me determine whether the following two answers to the quesiton are mathematical equivalent, ignoring any errors that might be caused by the format and ignoring any numerical units\' difference.\nThe first answer: {generation}\nThe second answer: {ground_truth}\nPlease give your answer in the following format: 'Equivalent' or 'Not Equivalent'."
        response = client.chat.completions.create(model="gemini-2.0-flash", messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt},])
        decision = response.choices[0].message.content.rstrip("\n")
        if decision == 'Equivalent':
            return True
        elif decision == 'Not Equivalent':
            return False
        elif "Not Equivalent" in decision:
            return False
        elif "Equivalent" in decision:
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

## math_equivalence.py

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2
