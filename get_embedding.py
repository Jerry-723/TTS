from datasets import TTSDataset
import argparse
import pickle
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="math500")
    args = parser.parse_args()

    model = SentenceTransformer("/mnt/sharedata/ssd_large/common/LLMs/all-MiniLM-L6-v2/")

    dataset = TTSDataset(args.dataset, 0.5)

    questions = [data["question"] for data in dataset.dataset]
    embeddings = model.encode(questions)
    dict_embeddings = {question: embedding for question, embedding in zip(questions, embeddings)}
    with open(f"outputs_exp/embeddings/{args.dataset}.pkl", "wb") as f:
        pickle.dump(dict_embeddings, f)