from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset
import pdb
import json


def get_dataset(dataset_name):
    if dataset_name == "s1K":
        dataset = load_from_disk("/mnt/sharedata/ssd2/users/zhanghx/dataset/s1K")
        dataset = dataset['train']
    
    elif dataset_name.name == "math500":
        dataset = []
        with open("dataset/assets/math500.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    "question": data["problem"],
                    "solution": data["solution"],
                    "answer": data["answer"]
                })
        dataset = Dataset.from_list(dataset)
        
    else:
        raise ValueError("Invalid dataset name")
    
    return dataset
    

def eval():
    data = get_dataset("s1K")    
    
    model = LLM(
        model = "/mnt/sharedata/ssd2/users/zhanghx/s1-32B",
        tensor_parallel_size=4, #
        gpu_memory_utilization=0.9,
    )
    tok = AutoTokenizer.from_pretrained(
        "/mnt/sharedata/ssd2/users/zhanghx/s1-32B"
    )


    stop_token_ids = tok("<|im_end|>")["input_ids"]

    sampling_params = SamplingParams(
        max_tokens=32768,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        temperature=0.0, # 0.0 means deterministic
    )

    for example in data:
        # question = example['question']
        question = "How many r in raspberry"
        prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
        outputs = model.generate(prompt, sampling_params=sampling_params)
        print(outputs[0].outputs[0].text)
        pdb.set_trace()


    
if __name__ == "__main__":
    eval()