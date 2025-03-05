from models import *
from datasets import *
from predictor import *
import argparse
import os
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500"], default="aime")
    parser.add_argument("--model", type=str, choices=["s1-32B"], default="s1-32B")
    parser.add_argument("--cal_size", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args()

    dataset = TTSDataset(name=args.dataset, percent=args.cal_size)
    inference = LLMInference(args.model)

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01"
    )

    predictor = conformalTTS(inference, args.alpha, dataset, client=client)
    predictor.predict()