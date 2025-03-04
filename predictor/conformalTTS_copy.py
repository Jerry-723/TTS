### This is a transformers version of conformalTTS, which is too slow for generate text.

# import numpy as np
# from common.scoring import ppl_score
# from models.utils import *
# # from vllm import SamplingParams
# import json
# from tqdm import tqdm

# MAX_TOKENS_THINKING = 32000

# class conformalTTS:
#     def __init__(self, LLMInference, alpha, dataset):
#         self.model = LLMInference.model
#         self.tokenizer = LLMInference.tokenizer
#         self.template = LLMInference.template
#         self.alpha = alpha
#         self.dataset = dataset

#     def calibrate(self):
#         cal_dataset = self.dataset.cal_dataset
#         n = len(cal_dataset)
#         scores = []
#         print("Calibrating...")
#         for sample in tqdm(cal_dataset):
#             prompt = self.template(sample['question']) + sample['solution']
#             score = ppl_score(self.model, self.tokenizer, prompt)
#             scores.append(score)
#         q_level = np.ceil((n + 1)*(1 - self.alpha))/n
#         tau = np.quantile(scores, q_level, method='higher')
#         return tau
    
#     def final_answer(self, text):
#         stop_token_ids = self.tokenizer("<|im_end|>")["input_ids"]
#         prompt = text + "Final Answer:"
#         _, new_text = self.generate(prompt, 32768,stop_token_ids=stop_token_ids)
#         return new_text
    
#     def generate(self, prompt, max_tokens, stop_token_ids):
#         stop_criteria = KeywordsStoppingCriteria(stop_token_ids)
#         input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 input_ids = input_ids.input_ids,
#                 attention_mask = input_ids.attention_mask,
#                 max_new_tokens=max_tokens,
#                 min_new_tokens=0,
#                 stopping_criteria=StoppingCriteriaList([stop_criteria]),
#                 temperature=None,
#                 top_k=None,
#                 top_p=None,
#             )
#         generated_new_tokens = outputs[0][len(input_ids[0]):-1]
#         generated_new_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):-1], skip_special_tokens=False)
#         return generated_new_tokens, generated_new_text
    
#     def predict(self):
#         test_dataset = self.dataset.test_dataset

#         tau = self.calibrate()
#         stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]

#         print("Predicting...")
#         for sample in tqdm(test_dataset):
#             answer_set = [] ## to be modify as prediction set
#             thinking_trace = ""

#             prompt = self.template(sample['question']) + "<|im_start|>think"
#             new_tokens, new_text = self.generate(prompt, MAX_TOKENS_THINKING, stop_token_ids=stop_token_ids)
#             ignore_str = "Wait"
#             max_tokens_thinking_tmp = MAX_TOKENS_THINKING - len(new_tokens)
#             while max_tokens_thinking_tmp > 0:
#                 prompt += new_text
#                 thinking_trace += new_text

#                 answer = self.final_answer(prompt)
#                 answer_set.append(answer)

#                 score = ppl_score(self.model, self.tokenizer, prompt)
#                 if score < tau:
#                     thinking_trace += "Final Answer:" + answer 
#                     break

#                 prompt += ignore_str
#                 thinking_trace += ignore_str

#                 new_tokens, new_text = self.generate(prompt, max_tokens_thinking_tmp, stop_token_ids=stop_token_ids)

#                 max_tokens_thinking_tmp -= len(new_tokens)
#             if max_tokens_thinking_tmp == 0:
#                 prompt += new_text
#                 answer = self.final_answer(prompt)
#                 answer_set.append(answer)
#                 thinking_trace += new_text + "Final Answer:" + answer

#                 score = ppl_score(self.model, self.tokenizer, prompt)
#                 if score > tau:
#                     answer_set.append("Non of above")
#             with open(f'/data/home/jiaxi/home/TTS/outputs/{self.dataset.name}_results.jsonl', 'a') as f:
#                 json.dump({"Question": sample['question'], "Thinking trace": thinking_trace, "Predicted set": answer_set, "Ground truth": sample['answer']}, f)
#                 f.write("\n")