from common import remove_duplicate_sentences
import json

if __name__=="__main__":
    with open(f"outputs_exp/s1.1-3B_math500_fullthinking.jsonl", "r") as f:
        lines = f.readlines()
        questions = [json.loads(line)["question"] for line in lines]
        ground_truths = [json.loads(line)["ground_truth"] for line in lines]
        responses = [json.loads(line)["response"] for line in lines]
        responses = [remove_duplicate_sentences(response, n=5) for response in responses]
    with open(f"outputs_exp/s1.1-3B_math500_fullthinking_nodup.jsonl", "w", encoding="utf-8") as f:
        for q, gd, r in zip(questions, ground_truths, responses):
            f.write(json.dumps({"question": q, "ground_truth": gd, "response": r}, ensure_ascii=False) + "\n")
