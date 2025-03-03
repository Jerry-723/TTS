# import numpy as np
# from common.scoring import ppl_score
# from vllm import SamplingParams
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
#         sampling_params = SamplingParams(
#             max_tokens=32768,
#             min_tokens=0,
#             stop_token_ids=stop_token_ids,
#             skip_special_tokens=False,
#             temperature=0.0
#         )
#         prompt = text + "Final Answer:"
#         o = self.model.generate(
#             prompt,
#             sampling_params=sampling_params
#         )
#         return o[0].outputs[0].text
    
#     def predict(self):
#         test_dataset = self.dataset.test_dataset

#         # tau = self.calibrate()
#         tau = 11.835315875876537 ## for aime
#         # tau = 12.803790605740799 ## for math500
#         stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
#         sampling_params = SamplingParams(
#             max_tokens=MAX_TOKENS_THINKING,
#             min_tokens=0,
#             stop_token_ids=stop_token_ids,
#             skip_special_tokens=False,
#             temperature=0.0
#         )
#         print("Predicting...")
#         for sample in tqdm(test_dataset):
#             answer_set = [] ## to be modify as prediction set
#             thinking_trace = ""

#             prompt = self.template(sample['question']) + "<|im_start|>think"
#             o = self.model.generate(
#                 prompt,
#                 sampling_params=sampling_params
#             )
#             ignore_str = "Wait"
#             max_tokens_thinking_tmp = MAX_TOKENS_THINKING - len(o[0].outputs[0].token_ids)
#             while max_tokens_thinking_tmp > 0:
#                 prompt += o[0].outputs[0].text
#                 thinking_trace += o[0].outputs[0].text

#                 answer = self.final_answer(prompt)
#                 answer_set.append(answer)

#                 score = ppl_score(self.model, self.tokenizer, prompt)
#                 if score < tau:
#                     thinking_trace += "Final Answer:" + answer 
#                     break

#                 prompt += ignore_str
#                 thinking_trace += ignore_str

#                 sampling_params = SamplingParams(
#                     max_tokens=max_tokens_thinking_tmp,
#                     min_tokens=0,
#                     stop_token_ids=stop_token_ids,
#                     skip_special_tokens=False,
#                     temperature=0.0
#                 )
#                 o = self.model.generate(
#                     prompt,
#                     sampling_params=sampling_params
#                 )
#                 max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
#             if max_tokens_thinking_tmp == 0:
#                 prompt += o[0].outputs[0].text
#                 answer = self.final_answer(prompt)
#                 answer_set.append(answer)
#                 thinking_trace += o[0].outputs[0].text + "Final Answer:" + answer

#                 score = ppl_score(self.model, self.tokenizer, prompt)
#                 if score > tau:
#                     answer_set.append("Non of above")
#             with open(f'/data/home/jiaxi/home/TTS/outputs/{self.dataset.name}_results.jsonl', 'a') as f:
#                 json.dump({"Question": sample['question'], "Thinking trace": thinking_trace, "Predicted set": answer_set, "Ground truth": sample['answer']}, f)
#                 f.write("\n")