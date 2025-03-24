from vllm import SamplingParams
from tqdm import tqdm
import json
import math
from common import is_equiv, extract_s1_answer

THINKING_GAP=100

class EfficientPred:
    def __init__(self, LLMInference, dataset, partial, batch_size):
        self.LLMInference = LLMInference
        self.model = LLMInference.model
        self.tokenizer = LLMInference.tokenizer
        self.template = LLMInference.template

        self.dataset = dataset
        self.partial = partial
        self.per_partial = math.ceil(len(self.dataset.dataset)/8)
        self.sub = self.dataset.dataset[partial * self.per_partial:(partial+1) * self.per_partial]

        self.batch_size = batch_size
        self.total_batch = math.ceil(len(self.sub)/self.batch_size)

    def forcefull(self, budget):
        stop_token_ids = self.tokenizer("<|im_start|><|im_end|>")["input_ids"]
        for i in tqdm(range(self.total_batch)):
            batch = self.sub[i*self.batch_size:(i+1)*self.batch_size]
            prompts = [self.template(p["question"]) + "<|im_start|>think" for p in batch]
            questions = [p["question"] for p in batch]
            ground_truth = [p["answer"] for p in batch]
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=budget,
                min_tokens=0,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False
            )
            outputs = self.model.generate(
                prompts,
                sampling_params=sampling_params
            )
            responses = [o.outputs[0].text for o in outputs]
            token_uses = [len(o.outputs[0].token_ids) for o in outputs]
            indices_to_continue = [i for i, token_use in enumerate(token_uses) if token_use < budget]
            while len(indices_to_continue) > 0:
                sampling_params = [
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=budget - token_uses[j],
                        min_tokens=0,
                        stop_token_ids=stop_token_ids,
                        skip_special_tokens=False
                    ) for j in indices_to_continue
                ]
                prompts_continue = [prompts[j] + responses[j] + "Wait" for j in indices_to_continue]
                outputs = self.model.generate(
                    prompts_continue,
                    sampling_params=sampling_params
                )
                for j, o in zip(indices_to_continue, outputs):
                    responses[j] += o.outputs[0].text
                    token_uses[j] += len(o.outputs[0].token_ids)
                indices_to_continue = [j for j, token_use in enumerate(token_uses) if token_use < budget]
            with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_fullthinking.jsonl", "a", encoding="utf-8") as f:
                for q, g, r in zip(questions, ground_truth, responses):
                    f.write(json.dumps({"question": q, "ground_truth": g, "response": r}, ensure_ascii=False) + "\n")

    def final_answer(self, prompt_token_ids):
        stop_token_ids = [self.tokenizer(text)["input_ids"][0] for text in ["\n\n", ".\n\n", ".\n", " \n\n", "<|im_end|>"]]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=500,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False
        )
        answer_token_ids = self.tokenizer("\n<|im_start|>answer\nFinal Answer: The final answer is $\\boxed{")["input_ids"]
        query_token_ids = [p_ids + answer_token_ids for p_ids in prompt_token_ids]
        outputs = self.model.generate(
            prompt_token_ids = query_token_ids,
            sampling_params=sampling_params
        )
        gen_answers = [extract_s1_answer(o.outputs[0].text) for o in outputs]
        return gen_answers
    
    def min_tokens(self):
        with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_fullthinking_nodup.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
            questions = [json.loads(l)["question"] for l in lines][self.partial * self.per_partial:(self.partial+1) * self.per_partial]
            ground_truth = [json.loads(l)["ground_truth"] for l in lines][self.partial * self.per_partial:(self.partial+1) * self.per_partial]
            responses = [json.loads(l)["response"] for l in lines][self.partial * self.per_partial:(self.partial+1) * self.per_partial]

        for i in tqdm(range(self.total_batch)):
            question_batch = questions[i*self.batch_size:(i+1)*self.batch_size]
            ground_truth_batch = ground_truth[i*self.batch_size:(i+1)*self.batch_size]
            responses_batch = responses[i*self.batch_size:(i+1)*self.batch_size]

            prompts = [self.template(q) + "<|im_start|>think" for q in question_batch]
            prompts_token_ids = self.tokenizer(prompts)["input_ids"]
            responses_token_ids = self.tokenizer(responses_batch)["input_ids"]

            thinking_budget = [len(r) for r in responses_token_ids]
            token_used = [0] * len(prompts)

            indices_to_continue = [j for j in range(len(prompts)) if token_used[j] < thinking_budget[j]]

            current_answers = [None] * len(prompts)

            while len(indices_to_continue) > 0:
                for j in indices_to_continue:
                    token_used[j] += THINKING_GAP
                query_token_ids = [prompts_token_ids[j] + responses_token_ids[j][:token_used[j]] for j in indices_to_continue]
                intermediate_responses = self.final_answer(query_token_ids)
                for j, inter_res in zip(indices_to_continue, intermediate_responses):
                    current_answers[j] = inter_res
                indices_to_continue = [j for j, r in zip(indices_to_continue, intermediate_responses) if not is_equiv(r, ground_truth_batch[j])]
                indices_to_continue = [j for j in indices_to_continue if token_used[j] < thinking_budget[j]]

            with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_min_tokens_rough.jsonl", "a", encoding="utf-8") as f:
                for q, g, r, budget, token, res_t in zip(question_batch, ground_truth_batch, current_answers, thinking_budget, token_used, responses_token_ids):
                    f.write(json.dumps({"question": q, "ground_truth": g, "token_used": token, "thinking_budget": budget, "generated_answer": r, "thinking": self.tokenizer.decode(res_t[:token])}, ensure_ascii=False) + "\n")
