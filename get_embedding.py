from vllm import LLM
from datasets import TTSDataset
from models import pretrained_model_dic
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="math500")
    parser.add_argument("--model", type=str, choices=["s1-32B", "DeepSeek-R1-Distill-Qwen-1.5B", "s1.1-3B", "s1.1-7B", "s1.1-14B"], default="s1.1-3B")
    args = parser.parse_args()

    dataset = TTSDataset(args.dataset, 0.5)

    model = LLM(pretrained_model_dic[args.model],
                tensor_parallel_size=2,
                task="embed")
    questions = [data["question"] for data in dataset.dataset]
    embeddings = model.embed(questions)
    dict_embeddings = {question: embedding.outputs.embedding for question, embedding in zip(questions, embeddings)}
    with open(f"outputs_exp/embeddings/{args.model}_{args.dataset}.pkl", "wb") as f:
        pickle.dump(dict_embeddings, f)