### This is a VLLM version of conformalTTS, which is faster for generate text.

import numpy as np
from common.scoring import ppl_score
from common.utils import extract_answer, judge_answer
from vllm import SamplingParams
import json
from tqdm import tqdm

MAX_TOKENS_THINKING = 32000
THINKNG_STEP_TOKENS = 500

class conformalTTS:
    def __init__(self, LLMInference, alpha, dataset, client):
        self.model = LLMInference.model
        self.tokenizer = LLMInference.tokenizer
        self.template = LLMInference.template
        self.alpha = alpha
        self.dataset = dataset
        self.client = client

    def calibrate(self):
        cal_dataset = self.dataset.cal_dataset
        n = len(cal_dataset)
        scores = []
        stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        print("Calibrating...")
        for sample in tqdm(cal_dataset):
            prompt = self.template(sample['question']) + "<|im_start|>think"
            thinking_trace = ""
            sampling_params = SamplingParams(
                max_tokens=MAX_TOKENS_THINKING,
                min_tokens=0,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.0
            )
            o = self.model.generate(
                prompt,
                sampling_params = sampling_params
            )
            ignore_str = "Wait"
            wait_count = 0
            max_tokens_thinking_tmp = MAX_TOKENS_THINKING - len(o[0].outputs[0].token_ids)
            answer_set = []
            while max_tokens_thinking_tmp >0:
                prompt += o[0].outputs[0].text
                thinking_trace += o[0].outputs[0].text
                answer = extract_answer(self.final_answer(prompt))
                answer_set.append(answer)
                if judge_answer(sample['question'], answer, sample['answer'], self.client):
                    break
                else:
                    prompt += ignore_str
                    thinking_trace += ignore_str
                    wait_count += 1

                    sampling_params = SamplingParams(
                        max_tokens=max_tokens_thinking_tmp,
                        min_tokens=0,
                        stop_token_ids=stop_token_ids,
                        skip_special_tokens=False,
                        temperature=0.0
                    )
                    o = self.model.generate(
                        prompt,
                        sampling_params = sampling_params
                    )
                    max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)

            if max_tokens_thinking_tmp == 0:
                prompt += o[0].outputs[0].text
                thinking_trace += o[0].outputs[0].text
            score = ppl_score(self.model, prompt)
            scores.append(score)
            token_use = MAX_TOKENS_THINKING - max_tokens_thinking_tmp

            with open(f"outputs/{self.dataset.name}_calibration.jsonl", "a") as f:
                json.dump({"Question": sample['question'], "Thinking trace": thinking_trace, "Token use": token_use, "Answer set": answer_set, "Ground truth": sample['answer'], "Wait count": wait_count, "PPL": score}, f)
                f.write("\n")

        q_level = np.ceil((n + 1)*(1 - self.alpha))/n
        tau = np.quantile(scores, q_level, method='higher')
        return tau

    def min_tokens(self):
        cal_dataset = self.dataset.cal_dataset
        stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        print("Calculating min tokens...")
        for sample in tqdm(cal_dataset):
            prompt = self.template(sample['question']) + "<|im_start|>think"
            token_used = 0
            while token_used < MAX_TOKENS_THINKING:
                sampling_params = SamplingParams(
                    max_tokens = THINKNG_STEP_TOKENS,
                    min_tokens = 0,
                    stop_token_ids = stop_token_ids,
                    skip_special_tokens = False,
                    temperature = 0.0
                )
                o = self.model.generate(
                    prompt,
                    sampling_params = sampling_params
                )
                prompt += o[0].outputs[0].text
                temp_left_tokens = THINKNG_STEP_TOKENS - len(o[0].outputs[0].token_ids)
                while temp_left_tokens > 1: ## 1 is the length of "Wait"
                    sampling_params = SamplingParams(
                        max_tokens = temp_left_tokens - 1,
                        min_tokens = 0,
                        stop_token_ids = stop_token_ids,
                        skip_special_tokens = False,
                        temperature = 0.0
                    )
                    prompt += "Wait"
                    o = self.model.generate(
                        prompt,
                        sampling_params = sampling_params
                    )
                    prompt += o[0].outputs[0].text
                    temp_left_tokens -= len(o[0].outputs[0].token_ids)
                token_used += THINKNG_STEP_TOKENS
                final_answer = self.final_answer(prompt)
                answer = extract_answer(final_answer)
                if judge_answer(sample['question'], answer, sample['answer'], self.client):
                    break
            with open(f"outputs/{self.dataset.name}_calibration_min_tokens.jsonl", "a") as f:
                json.dump({"Question": sample['question'], "Thinking trace": prompt, "Token use": token_used, "Predict answer": answer, "Ground truth": sample['answer']}, f)
                f.write("\n")

    def no_thinking(self):
        cal_dataset = self.dataset.cal_dataset
        stop_token_ids = self.tokenizer("\n\n<|im_end|>")["input_ids"]
        print("No thinking...")
        for sample in tqdm(cal_dataset):
            prompt = self.template(sample['question']) + "<|im_start|>answer\nFinal Answer: The final answer is"
            sampling_params = SamplingParams(
                max_tokens = MAX_TOKENS_THINKING,
                min_tokens = 0,
                stop_token_ids = stop_token_ids,
                skip_special_tokens = False,
                temperature = 0.0
            )
            o = self.model.generate(
                prompt,
                sampling_params = sampling_params
            )
            answer = extract_answer(o[0].outputs[0].text)
            with open(f"outputs/{self.dataset.name}_nothinking.jsonl", "a") as f:
                json.dump({"Question": sample['question'], "Answer": answer}, f)
                f.write("\n")

    def final_answer(self, text):
        """
        Get final answer from the template answer between \\boxed{}
        """
        stop_token_ids = self.tokenizer("<|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=32768,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0
        )
        prompt = text + "<|im_start|>answer\nFinal Answer:"
        o = self.model.generate(
            prompt,
            sampling_params=sampling_params
        )
        # answer = extract_answer(o[0].outputs[0].text)
        return o[0].outputs[0].text
    
    def predict(self):
        test_dataset = self.dataset.test_dataset
        tau = self.calibrate()
        stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0
        )
        print("Predicting...")
        for sample in tqdm(test_dataset):
            answer_set = []
            thinking_trace = ""

            prompt = self.template(sample['question']) + "<|im_start|>think"
            o = self.model.generate(
                prompt,
                sampling_params=sampling_params
            )
            ignore_str = "Wait"
            max_tokens_thinking_tmp = MAX_TOKENS_THINKING - len(o[0].outputs[0].token_ids)
            while max_tokens_thinking_tmp > 0:
                prompt += o[0].outputs[0].text
                thinking_trace += o[0].outputs[0].text

                answer = self.final_answer(prompt)
                answer_set.append(extract_answer(answer))

                score = ppl_score(self.model, prompt)
                if score < tau:
                    thinking_trace += "<|im_start|>answer\nFinal Answer:" + answer 
                    break

                prompt += ignore_str
                thinking_trace += ignore_str

                sampling_params = SamplingParams(
                    max_tokens=max_tokens_thinking_tmp,
                    min_tokens=0,
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=0.0
                )
                o = self.model.generate(
                    prompt,
                    sampling_params=sampling_params
                )
                max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            if max_tokens_thinking_tmp == 0:
                prompt += o[0].outputs[0].text
                answer = self.final_answer(prompt)
                answer_set.append(extract_answer(answer))
                thinking_trace += o[0].outputs[0].text + "<|im_start|>answer\nFinal Answer:" + answer

                score = ppl_score(self.model, prompt)
                if score > tau:
                    answer_set.append("Non of above")
            with open(f'outputs/{self.dataset.name}_results.jsonl', 'a') as f:
                json.dump({"Question": sample['question'], "Thinking trace": thinking_trace, "Predicted set": answer_set, "Ground truth": sample['answer'], "PPL": score, "tau": tau}, f)
                f.write("\n")

    def tau_tokens(self):
        """
        用一次得到正确结果的tokens的90%分位数作为tau的尝试
        """
        tokens = []
        name = self.dataset.name
        file_path = f"outputs/{name}_calibration.jsonl"
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data['Token use'] != 32000 and data['Wait count'] == 0:
                    tokens.append(data['Token use'])
        q_level = np.ceil((len(tokens) + 1)*(1 - self.alpha))/len(tokens)
        tau = np.quantile(tokens, q_level, method='higher')
        return tau

    def budget_force_predict(self):
        test_dataset = self.dataset.test_dataset
        tau_token = self.tau_tokens()
        stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=tau_token,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0
        )
        print("Budget force predicting...")
        for sample in tqdm(test_dataset):
            answer_set = []
            thinking_trace = ""

            prompt = self.template(sample['question']) + "<|im_start|>think"
            o = self.model.generate(
                prompt,
                sampling_params=sampling_params
            )
            ignore_str = "Wait"
            max_budget_tokens_tmp = tau_token - len(o[0].outputs[0].token_ids)
            while max_budget_tokens_tmp > 0:
                prompt += o[0].outputs[0].text
                thinking_trace += o[0].outputs[0].text

                answer = self.final_answer(prompt)
                answer_set.append(extract_answer(answer))

                prompt += ignore_str
                thinking_trace += ignore_str

                sampling_params = SamplingParams(
                    max_tokens=max_budget_tokens_tmp,
                    min_tokens=0,
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=0.0
                )
                o = self.model.generate(
                    prompt,
                    sampling_params=sampling_params
                )
                max_budget_tokens_tmp -= len(o[0].outputs[0].token_ids)
            if max_budget_tokens_tmp == 0:
                prompt += o[0].outputs[0].text
                answer = self.final_answer(prompt)
                answer_set.append(extract_answer(answer))
                thinking_trace += o[0].outputs[0].text + "<|im_start|>answer\nFinal Answer:" + answer
            with open(f'outputs/{self.dataset.name}_results_budget_force.jsonl', 'a') as f:
                json.dump({"Question": sample['question'], "Thinking trace": thinking_trace, "Predicted set": answer_set, "Ground truth": sample['answer'], "tau": int(tau_token)}, f)
                f.write("\n")