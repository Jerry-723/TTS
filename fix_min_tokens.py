from common import judge_answer, is_equiv
from models import pretrained_model_dic
import json
import argparse
from openai import AzureOpenAI
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import math

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="math500")
    parser.add_argument("--model", type=str, choices=["s1-32B", "DeepSeek-R1-Distill-Qwen-1.5B", "s1.1-3B", "s1.1-7B", "s1.1-14B"], default="s1.1-3B")
    parser.add_argument("--partial", type=int, default=0)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(pretrained_model_dic[args.model])

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01"
    )
    
    with open(f"outputs_exp/{args.model}_{args.dataset}_min_tokens_rough.jsonl", "r") as f:
        lines = f.readlines()
        per_partial = math.ceil(len(lines)/10)
        start = 0
        questions = [json.loads(line)["question"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        ground_truths = [json.loads(line)["ground_truth"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        generated_answers = [json.loads(line)["generated_answer"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        answer_traces = [json.loads(line)["answer_trace"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        thinking_budgets = [json.loads(line)["thinking_budget"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        token_used = [json.loads(line)["token_used"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        responses = [json.loads(line)["thinking"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]

    with open(f"outputs_exp/{args.model}_{args.dataset}_min_tokens_fixed.jsonl", "a", encoding="utf-8") as f:
        for q, gd, ga, at, tb, tu, r in tqdm(zip(questions, ground_truths, generated_answers, answer_traces, thinking_budgets, token_used, responses), total=len(questions), desc="Processing Samples", unit="sample"):
            # if tu >= tb:
            flag = "Wrong"
            wrong_answers = []
            for i, at_i in enumerate(at):
                if at_i not in wrong_answers:
                    if is_equiv(at_i, gd):
                        flag = "Correct"
                        break
                    elif judge_answer(q, at_i, gd, client):
                        flag = "Correct"
                        break
                    elif judge_answer(q, at_i, gd, client): ## twice check
                        flag = "Correct"
                        break
                    else:   
                        wrong_answers.append(at_i)
            ga = at_i
            at = at[:(i+1)]
            tu = (i+1)*100
            token_ids = tok(r)["input_ids"]
            r = tok.decode(token_ids[:tu])

            f.write(json.dumps({"question": q, "ground_truth": gd, "generated_answer": ga, "answer_trace": at, "thinking_budget": tb, "token_used": tu, "flag": flag, "thinking": r}, ensure_ascii=False) + "\n")
            print("done")

        print("Finished!")
