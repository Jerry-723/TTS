from common import judge_answer
from models import pretrained_model_dic
import json
import argparse
from tqdm import tqdm
import math

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="math500")
    parser.add_argument("--model", type=str, choices=["s1-32B", "DeepSeek-R1-Distill-Qwen-1.5B", "s1.1-3B", "s1.1-7B", "s1.1-14B"], default="s1.1-3B")
    parser.add_argument("--partial", type=int, default=0)
    args = parser.parse_args()
    
    with open(f"outputs_exp/{args.model}_{args.dataset}_fulltrace.jsonl", "r") as f:
        lines = f.readlines()
        per_partial = math.ceil(len(lines)/5)
        start = 0
        questions = [json.loads(line)["question"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        ground_truths = [json.loads(line)["ground_truth"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        answer_traces = [json.loads(line)["answer_trace"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        thinking_budgets = [json.loads(line)["thinking_budget"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        token_used = [json.loads(line)["token_used"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]
        responses = [json.loads(line)["thinking"] for line in lines][(args.partial * per_partial+start):(args.partial + 1)*per_partial]

    with open(f"outputs_exp/{args.model}_{args.dataset}_fulltrace_withrange.jsonl", "a", encoding="utf-8") as f:
        for q, gd, at, tb, tu, r in tqdm(zip(questions, ground_truths, answer_traces, thinking_budgets, token_used, responses), total=len(questions), desc="Processing Samples", unit="sample"):
            flag = "Wrong"
            wrong_answers = []
            right_answers = []
            correct_index = []
            for i, at_i in enumerate(at):
                if at_i not in wrong_answers:
                    if at_i in right_answers:
                        correct_index.append(i+1)
                    elif judge_answer(q, at_i, gd):
                        flag = "Correct"
                        right_answers.append(at_i)
                        correct_index.append(i+1)
                    elif judge_answer(q, at_i, gd): ## twice check
                        flag = "Correct"
                        right_answers.append(at_i)
                        correct_index.append(i+1)
                    else:   
                        wrong_answers.append(at_i)

            f.write(json.dumps({"question": q, "ground_truth": gd, "answer_trace": at, "correct_index": correct_index, "thinking_budget": tb, "token_used": tu, "flag": flag, "thinking":r}, ensure_ascii=False) + "\n")
            print("done")

        print("Finished!")
