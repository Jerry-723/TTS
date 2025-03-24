from common import remove_duplicate_sentences
import json
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="math500")
    parser.add_argument("--model", type=str, choices=["s1-32B", "DeepSeek-R1-Distill-Qwen-1.5B", "s1.1-3B", "s1.1-7B", "s1.1-14B"], default="s1.1-3B")
    args = parser.parse_args()
    
    with open(f"outputs_exp/{args.model}_{args.dataset}_fullthinking.jsonl", "r") as f:
        lines = f.readlines()
        questions = [json.loads(line)["question"] for line in lines]
        ground_truths = [json.loads(line)["ground_truth"] for line in lines]
        responses = [json.loads(line)["response"] for line in lines]
        responses = [remove_duplicate_sentences(response, n=5) for response in responses]
    with open(f"outputs_exp/{args.model}_{args.dataset}_fullthinking_nodup.jsonl", "w", encoding="utf-8") as f:
        for q, gd, r in zip(questions, ground_truths, responses):
            f.write(json.dumps({"question": q, "ground_truth": gd, "response": r}, ensure_ascii=False) + "\n")
