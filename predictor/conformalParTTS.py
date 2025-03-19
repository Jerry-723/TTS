from vllm import SamplingParams
from common.utils import extract_qwen_answer, judge_answer
import numpy as np
import json
from tqdm import tqdm
import re
import os
import pickle

PREDICT_BUDGET = 10

class conformalParTTS:
    def __init__(self, LLMInference, cal_set, pred_set, batch_size, partial, client):
        self.model = LLMInference.model
        self.tokenizer = LLMInference.tokenizer
        self.template = LLMInference.template

        self.cal_set_name = cal_set.name
        self.cal_set = cal_set.dataset
        per_cal_partial = int(len(self.cal_set)/8)
        self.cal_set = self.cal_set[partial * per_cal_partial:(partial+1) * per_cal_partial]

        self.pred_set_name = pred_set.name
        self.pred_set = pred_set.dataset
        per_pred_partial = int(len(self.pred_set)/8)
        self.pred_set = self.pred_set[partial * per_pred_partial:(partial+1) * per_pred_partial]
        
        self.batch_size = batch_size
        self.client = client

    def get_conf(self, prompts):
        confidences = []
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20
        )
        prompts = [prompt + "<｜end▁of▁sentence｜><｜User｜>Directly answer the question by Yes or No:Is the answer correct?<｜Assistant｜></think>\n\n)" for prompt in prompts]
        o = self.model.generate(
            prompts,
            sampling_params=sampling_params
        )
        for output in o:
            logprobs = output.outputs[0].logprobs[0]
            true_prob = sum(np.exp(logprob.logprob) for token_id, logprob in logprobs.items() if 'Yes' in logprob.decoded_token)
            false_prob = sum(np.exp(logprob.logprob) for token_id, logprob in logprobs.items() if 'No' in logprob.decoded_token)
            confidence = true_prob/(true_prob+false_prob) if true_prob + false_prob > 0 else 0
            confidences.append(confidence)
        return confidences

    def calibrate(self):
        stop_token_ids = self.tokenizer("<｜end▁of▁sentence｜>")["input_ids"]
        total_batchs = int(len(self.cal_set)/self.batch_size) + 1
        for i in tqdm(range(total_batchs)):
            batch = self.cal_set[i*self.batch_size:(i+1)*self.batch_size]
            prompts = [self.template(p["question"]) for p in batch]
            questions = [p["question"] for p in batch]
            ground_truth = [p["answer"] for p in batch]
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=16384,
                min_tokens=0,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False
            )
            outputs = self.model.generate(
                prompts,
                sampling_params=sampling_params
            )
            responses = [o.outputs[0].text for o in outputs]

            thinkings = [re.search(r'^(.*?)</think>', r, re.DOTALL).group(1) if re.search(r'^(.*?)</think>', r, re.DOTALL) else r for r in responses]
            answers = [re.search(r'</think>(.*)', r, re.DOTALL).group(1) if re.search(r'</think>(.*)', r, re.DOTALL) else None for r in responses]
            extract_answers = [extract_qwen_answer(a) for a in answers]

            conf_prompts = [o.prompt + o.outputs[0].text for o in outputs]
            confs = self.get_conf(conf_prompts)
            with open(f"outputs/{self.cal_set_name}_calibration_par.jsonl", "a", encoding="utf-8") as f:
                for q, t, a, ea, conf, gt in zip(questions, thinkings, answers, extract_answers, confs, ground_truth):
                    json.dump({"Question": q, "Thinking": t, "Answer": a, "Extract answer": ea, "Ground truth": gt, "Confidence": conf}, f, ensure_ascii=False)
                    f.write("\n")

    def get_confs(self):
        confs = []
        file_path_root = f"outputs/{self.cal_set_name}_"
        if not os.path.exists(f"{file_path_root}conf.pkl"):
            if not os.path.exists(f"{file_path_root}calibration_par.jsonl"):
                self.calibrate()
            with open(f"{file_path_root}calibration_par.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data["Extract answer"] == data["Ground truth"]:
                        confs.append(data["Confidence"])
                    elif data["Answer"] != None:
                        if data["Extract answer"] != None:
                            generation = data["Extract answer"]
                        else:
                            generation = data["Answer"]
                        if judge_answer(data["Question"], generation, data["Ground truth"], self.client):
                            confs.append(data["Confidence"])
            with open(f"{file_path_root}conf.pkl", "wb") as f:
                pickle.dump(confs, f)
        else:
            with open(f"{file_path_root}conf.pkl", "rb") as f:
                confs = pickle.load(f)
        return confs

    def get_thr(self, alpha):
        confs = self.get_confs()
        scores = [1-conf for conf in confs]
        n = len(scores)
        q_level = np.ceil((n + 1)*(1-alpha))/n
        tau = np.quantile(scores, q_level, method="higher")
        thr = 1 - tau
        print(f"Threshold on {self.cal_set_name} of alpha {alpha} is {thr}.")
        return thr
        
    def predict(self, alpha):
        thr = self.get_thr(alpha)
        stop_token_ids = self.tokenizer("<｜end▁of▁sentence｜>")["input_ids"]
        total_batchs = int(len(self.pred_set)/self.batch_size) + 1
        for i in tqdm(range(total_batchs)):
            batch = self.pred_set[i*self.batch_size:(i+1)*self.batch_size]
            prompts = [self.template(p["question"]) for p in batch]
            questions = [p["question"] for p in batch]
            ground_truth = [p["answer"] for p in batch]
            sampling_params = SamplingParams(
                temperature=1.0,
                max_tokens=16384,
                min_tokens=0,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                n=10
            )
            pred_sets = []
            outputs = self.model.generate(
                prompts,
                sampling_params=sampling_params
            )
            for output in outputs:
                responses = [o.text for o in output.outputs]
                answers = [re.search(r'</think>(.*)', r, re.DOTALL).group(1) if re.search(r'</think>(.*)', r, re.DOTALL) else None for r in responses]
                extract_qwen_answers = [extract_qwen_answer(a) for a in answers]
                conf_prompts = [output.prompt + r for r in responses]
                confs = self.get_conf(conf_prompts)
                answer_set = [eqa for eqa, c in zip(extract_qwen_answers, confs) if c >= thr]
                pred_sets.append(answer_set)
            with open(f"outputs/{self.pred_set_name}_prediction_par_{PREDICT_BUDGET}_{alpha}.jsonl", "a", encoding="utf-8") as f:
                for q, gt, ps in zip(questions, ground_truth, pred_sets):
                    json.dump({"Question": q, "Prediction set": ps, "Ground truth": gt}, f, ensure_ascii=False)
                    f.write("\n")

    def get_coverage(self, alpha):
        with open(f"outputs/{self.pred_set_name}_prediction_par_{PREDICT_BUDGET}_{alpha}.jsonl", "r") as f:
            total = 0
            correct = 0
            for line in f:
                data = json.loads(line)
                total += 1
                if data["Ground truth"] in data["Prediction set"]:
                    correct += 1
                elif len(data["Prediction set"]) == 0:
                    correct += 1
                else:
                    for a in data["Prediction set"]:
                        if judge_answer(data["Question"], a, data["Ground truth"], self.client):
                            correct += 1
                            break
        print(f"Coverage on {self.pred_set_name} of alpha {alpha} is {correct/total}.")
        return correct/total