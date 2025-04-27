from vllm import SamplingParams
from tqdm import tqdm
import json
import math
from common import is_equiv, extract_s1_answer, judge_answer
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

THINKING_GAP=100

class EfficientPred:
    def __init__(self, LLMInference, dataset, partial, batch_size, alpha):
        self.LLMInference = LLMInference
        self.model = LLMInference.model
        self.tokenizer = LLMInference.tokenizer
        self.template = LLMInference.template

        self.dataset = dataset

        self.cal_set, self.test_set = self.dataset.cal_dataset, self.dataset.test_dataset

        self.partial = partial
        self.per_partial = math.ceil(len(self.dataset.dataset)/2)
        self.sub = self.dataset.dataset[partial * self.per_partial:(partial+1) * self.per_partial]

        self.batch_size = batch_size
        self.total_batch = math.ceil(len(self.sub)/self.batch_size)
        self.alpha = alpha

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
    
    def nodup_answer(self, client):
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
            query_token_ids = [prompts_token_ids[j] + responses_token_ids[j] for j in range(len(prompts))]
            final_answers = self.final_answer(query_token_ids)
            flags = ["Wrong"]*len(prompts)
            for j, (fa, gt) in enumerate(zip(final_answers, ground_truth_batch)):
                if is_equiv(fa, gt):
                    flags[j] = "Correct"
                elif judge_answer(question_batch[j], fa, gt, client):
                    flags[j] = "Correct"
                elif judge_answer(question_batch[j], fa, gt, client):
                    flags[j] = "Correct"
            with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_nodup_withans.jsonl", "a", encoding="utf-8") as f:
                for q, g, r, fa, flag in zip(question_batch, ground_truth_batch, responses_batch, final_answers, flags):
                    f.write(json.dumps({"question": q, "ground_truth": g, "response": r, "final_answer": fa, "flag": flag}, ensure_ascii=False) + "\n")
    
    def full_trace(self, gpt=False, client=None):
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
            answer_traces = [[] for _ in range(len(prompts))]

            indices_to_continue = [j for j in range(len(prompts)) if token_used[j] < thinking_budget[j]]

            while len(indices_to_continue) > 0:
                for j in indices_to_continue:
                    token_used[j] += THINKING_GAP
                query_token_ids = [prompts_token_ids[j] + responses_token_ids[j][:token_used[j]] for j in indices_to_continue]
                intermediate_responses = self.final_answer(query_token_ids)
                for j, inter_res in zip(indices_to_continue, intermediate_responses):
                    answer_traces[j].append(inter_res)
                # if gpt == True:
                #     indices_to_continue = [j for j, r in zip(indices_to_continue, intermediate_responses) if not is_equiv(r, ground_truth_batch[j]) and not judge_answer(question_batch[j], r, ground_truth_batch[j], client)]
                # else:
                #     indices_to_continue = [j for j, r in zip(indices_to_continue, intermediate_responses) if not is_equiv(r, ground_truth_batch[j])]
                indices_to_continue = [j for j in indices_to_continue if token_used[j] < thinking_budget[j]]

            with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_fulltrace.jsonl", "a", encoding="utf-8") as f:
                for q, g, ans_t, budget, token, res_t in zip(question_batch, ground_truth_batch, answer_traces, thinking_budget, token_used, responses_token_ids):
                    f.write(json.dumps({"question": q, "ground_truth": g, "token_used": token, "thinking_budget": budget, "answer_trace": ans_t, "thinking": self.tokenizer.decode(res_t[:token])}, ensure_ascii=False) + "\n")
    
    # def calibrate(self, beta=0.1):
    #     cal_questions = [data["question"] for data in self.cal_set]
    #     ks = []
    #     with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_ppl.pkl", "rb") as f:
    #         cal_ppl = pickle.load(f)
    #     with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_min_tokens_fixed.jsonl", "r", encoding="utf-8") as f:
    #         lines = f.readlines()
    #         for l in lines:
    #             line = json.loads(l)
    #             q = line["question"]
    #             if q in cal_questions:
    #                 if line["flag"] == "Correct":
    #                     resp_start = cal_ppl[q]['resp_start']
    #                     token_used = line['token_used']
    #                     ppls = cal_ppl[q]['ppl']
    #                     k = np.exp((ppls[resp_start+token_used-1] - ppls[resp_start+token_used-101])/100)
    #                     ks.append(k)
    #     n = len(ks)
    #     q_level = np.ceil((n + 1)*(1 - beta))/n
    #     thr = np.quantile(ks, q_level, method="higher")
    #     return thr
    
    # def predict(self, client):
    #     # thr = self.calibrate()
    #     # pred_questions = [data["question"] for data in self.test_set]
    #     pred_questions = [data["question"] for data in self.cal_set]
    #     with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_fullthinking_nodup.jsonl", "r") as f:
    #         lines = f.readlines()
    #         questions = [json.loads(l)["question"] for l in lines]
    #         ground_truth = [json.loads(l)["ground_truth"] for l in lines]
    #         responses = [json.loads(l)["response"] for l in lines]
    #     with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_ppl.pkl", "rb") as f:
    #         pred_ppl = pickle.load(f)
    #     with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_min_tokens_fixed.jsonl", "r") as f:
    #         lines = f.readlines()
    #         for l in lines:
    #             line = json.loads(l)
    #             q = line['question']
    #             if q in pred_questions:
    #                 if line['flag'] == "Wrong":
    #                     pred_questions.remove(q)

    #     for i in np.linspace(0.01,0.15,15):
    #         thr = self.calibrate(i)
    #         thr = 1
    #         print("Current thr: ", thr, "Current alpha: ", i)
    #         budgets = []
    #         for q in tqdm(questions, desc="Predicting token budget:"):
    #             if q in pred_questions:
    #                 resp_start = pred_ppl[q]['resp_start']
    #                 ppls = pred_ppl[q]['ppl']
    #                 ks = [np.exp((ppls[resp_start+i*100-1] - ppls[resp_start+i*100-101])/100) for i in range(1, int((len(ppls)-resp_start)/100))]
    #                 budget = None
    #                 for i in range(1, int((len(ppls)-resp_start)/100)):
    #                     if ks[i-1] > thr:
    #                         budget = i*100
    #                         break
    #                 if budget is None:
    #                     budget = len(ppls) - resp_start
    #                 budgets.append(budget)
    #         prompts = [self.template(q) + "<|im_start|>think" for q in pred_questions]
    #         prompts_token_ids = self.tokenizer(prompts)["input_ids"]
    #         responses_token_ids = self.tokenizer(responses)["input_ids"]
    #         query_token_ids = [prompts_token_ids[j] + responses_token_ids[j][:budgets[j]] for j in range(len(prompts))]
    #         final_answers = self.final_answer(query_token_ids)
    #         flags = []
    #         for q, f, gd in zip(pred_questions, final_answers, ground_truth):
    #             flag = "Wrong"
    #             if is_equiv(f, gd):
    #                 flag = "Correct"
    #             # elif judge_answer(q, f, gd, client):
    #             #     flag = "Correct"
    #             # elif judge_answer(q, f, gd, client):
    #             #     flag = "Correct"
    #             flags.append(flag)

    #         print("Coverage: ", len([f for f in flags if f == "Correct"])/len(flags))
            
    #     # with open(f"result/Calibrate_{self.LLMInference.name}_{self.dataset.name}_final.jsonl", "w", encoding="utf-8") as f:
    #     #     for q, g, b, fa, flag in zip(pred_questions, ground_truth, budgets, final_answers, flags):
    #     #         f.write(json.dumps({"question": q, "ground_truth": g, "budget": b, "final_answer": fa, "flag": flag}, ensure_ascii=False) + "\n")

    def is_continuous(self, lst, conti_count=5):
        count = 1
        start_index = 0
        for i in range(1,len(lst)):
            if lst[i] == lst[i-1]+1:
                if count == 1:
                    start_index = i - 1
                count += 1
                if count >= conti_count:
                    return lst[start_index]
            else:   
                count = 1
        return -1
    
    def calibrate_pred(self):
        with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_ppl.pkl", "rb") as f:
            ppls = pickle.load(f)
        cal_questions = [p["question"] for p in self.cal_set]
        pred_questions = [p["question"] for p in self.test_set]
        cal_ppls = {q: ppls[q] for q in cal_questions}
        pred_ppls = {q: ppls[q] for q in pred_questions}
        with open(f"outputs_exp/{self.LLMInference.name}_{self.dataset.name}_fulltrace_withrange.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
            questions = [json.loads(l)["question"] for l in lines]
            flags = [json.loads(l)["flag"] for l in lines]
            correct_indexs = [json.loads(l)["correct_index"] for l in lines]
            thinking_budgets = [json.loads(l)["thinking_budget"] for l in lines]
            token_useds = [json.loads(l)["token_used"] for l in lines]
            dictionary = {q: {"flag": f, "correct_index": ci, "token_used": tu, "thinking_budget": tb} for q, f, ci, tu, tb in zip(questions, flags, correct_indexs, token_useds, thinking_budgets)}
        with open(f"outputs_exp/embeddings/{self.dataset.name}.pkl", "rb") as f:
            embeddings = pickle.load(f)

        refine_cal = {}
        for q in cal_questions:
            if dictionary[q]["flag"] == "Correct":
                k_start = self.is_continuous(dictionary[q]["correct_index"])
                if k_start != -1:
                    resp_start = cal_ppls[q]['resp_start']
                    token_used = k_start * 100
                    ppl = cal_ppls[q]['ppl']
                    # k = np.exp((ppl[resp_start+token_used-1] - ppl[resp_start+token_used-101])/100)
                    k = (ppl[resp_start+token_used-1] - ppl[resp_start+token_used-101])/100
                    start_ppl = ppl[resp_start-1]
                    refine_cal[q] = {'k': k, "start_ppl": start_ppl}
        
        refine_cal_qs = list(refine_cal.keys())
        refine_cal_ks = [refine_cal[q]['k'] for q in refine_cal_qs]
        refine_cal_start_ppls = [refine_cal[q]['start_ppl'] for q in refine_cal_qs]

        cal_embeddings = {q: embeddings[q] for q in refine_cal_qs}
        cal_embedding_matrix = np.array(list(cal_embeddings.values()))

        count = 0
        token_saved = []

        for q in pred_questions:
            pred_embed = embeddings[q]
            similarities = cosine_similarity([pred_embed], cal_embedding_matrix)[0]
            top_n_indices = np.argsort(similarities)[-6:-1][::-1]
            ks = []
            resp_start = pred_ppls[q]['resp_start']
            ppl = pred_ppls[q]['ppl']
            for i in range(1, int((len(ppl)-resp_start)/100)):
                # k = np.exp((ppl[resp_start+i*100-1] - ppl[resp_start+i*100-101])/100)
                k = (ppl[resp_start+i*100-1] - ppl[resp_start+i*100-101])/100
                ks.append(k)
            pred_ppls[q]['ks'] = ks
            start_ppl = ppl[resp_start-1]
            distances = np.abs(np.array(refine_cal_start_ppls)[top_n_indices] - start_ppl)
            closest_indices = np.argsort(distances)[:5]
            # closest_qs = [refine_cal_qs[i] for i in closest_indices]
            # closest_correct_indexes = [dictionary[q]["correct_index"][0] for q in closest_qs]
            # weights = [cci/sum(closest_correct_indexes) for cci in closest_correct_indexes]
            # closest_ks = [refine_cal_ks[i] for i in closest_indices]
            # k_avg = np.average(closest_ks, weights=weights)
            k_avg = max([refine_cal_ks[i] for i in closest_indices])

            pred_index = next((i for i, k in enumerate(ks) if k > k_avg), len(ks))

            if dictionary[q]["correct_index"] != []:
                if pred_index+1 >= dictionary[q]["correct_index"][0]:
                    count += 1

            actual_token_used = (pred_index+1) * 100
            token_saved.append((dictionary[q]["thinking_budget"] - actual_token_used)/dictionary[q]["thinking_budget"])

        pred_count = 0
        full_count = 0

        for q in pred_questions:
            if dictionary[q]["flag"] == "Correct":
                pred_count += 1
            if dictionary[q]["correct_index"] != [] and dictionary[q]["token_used"]/100 == dictionary[q]["correct_index"][-1]:
                full_count += 1
        print("Realization: ", pred_count/len(pred_questions))
        print("Full thinking: ", full_count/len(pred_questions))
        print("Accuracy: ", count/len(pred_questions))
        print("Token saved: ", np.mean(token_saved))