import argparse
from vllm import LLM, SamplingParams
from models import LLMInference
import json
import math
import pickle

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="math500")
    parser.add_argument("--model", type=str, choices=["s1-32B", "DeepSeek-R1-Distill-Qwen-1.5B", "s1.1-3B", "s1.1-7B", "s1.1-14B"], default="s1.1-3B")

    args = parser.parse_args()

    inference = LLMInference(args.model)
    sampling_params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1
    )
    with open(f"outputs_exp/{args.model}_{args.dataset}_fulltrace.jsonl", "r") as f:
        lines = f.readlines()
        questions = [json.loads(line)["question"] for line in lines]
        thinking_budgets = [json.loads(line)["thinking_budget"] for line in lines]
        responses = [json.loads(line)["thinking"] for line in lines]

    inputs = [inference.template(q) + "<|im_start|>think" + res for q, res in zip(questions, responses)]

    o = inference.model.generate(
        inputs,
        sampling_params=sampling_params
    )
    logprobs = [out.prompt_logprobs for out in o]
    lengths = [len(logprob) for logprob in logprobs]
    resp_starts = [length - tb for length, tb in zip(lengths, thinking_budgets)]
    ppls = []
    for logprob in logprobs:
        ppl = [None]
        log_prob_total = 0
        for index, i in enumerate(logprob):
            if i is not None:
                _, logp = next(iter(i.items()))
                log_prob_total += logp.logprob
                ppl.append(math.exp(-log_prob_total/(index)))
        ppls.append(ppl)

    dict_ppl = {q:{"ppl": p, "resp_start": rs} for q, p, rs in zip(questions, ppls, resp_starts)}
    with open(f"outputs_exp/{args.model}_{args.dataset}_ppl.pkl", "wb") as f:
        pickle.dump(dict_ppl, f)