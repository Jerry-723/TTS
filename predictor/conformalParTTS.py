from vllm import SamplingParams
import numpy as np
import json
from tqdm import tqdm

class conformalParTTS:
    def __init__(self, LLMInference, cal_set, pred_set, batch_size):
        self.model = LLMInference.model
        self.tokenizer = LLMInference.tokenizer
        self.template = LLMInference.template
        self.cal_set = cal_set
        self.pred_set = pred_set
        self.batch_size = batch_size

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
            confidence = true_prob/(true_prob+false_prob)
            confidences.append(confidence)
        return confidences

    def calibrate(self):
        stop_token_ids = self.tokenizer("<｜end▁of▁sentence｜>")["input_ids"]
        total_batchs = int(len(self.cal_set)/self.batch_size) + 1
        for i in tqdm(range(total_batchs)):
            batch = self.cal_set[i*self.batch_size:(i+1)*self.batch_size]
            prompts = [self.template(p["question"]) for p in batch]
            answers = [p["answer"] for p in batch]
            sampling_params = SamplingParams(
                temperature = 0.0,
                max_tokens = 32000,
                min_tokens = 0,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False
            )
            outputs = self.model.generate(
                prompts,
                sampling_params=sampling_params
            )
            conf_prompts = [o.prompt + o.outputs[0].text for o in outputs]
            confs = self.get_conf(conf_prompts)
            with open(f"outputs/{self.cal_set.name}_calibration_par.jsonl", "a") as f:
                for p, conf, a in zip(conf_prompts, confs, answers):
                    json.dump({"Response": p, "Ground truth": a, "Confidence": conf}, f)
                    f.write("\n")
