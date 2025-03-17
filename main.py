from models import *
from datasets import *
from predictor import *
import argparse
import os
from openai import AzureOpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="gsm8k")
    parser.add_argument("--model", type=str, choices=["s1-32B", "DeepSeek-R1-Distill-Qwen-1.5B"], default="DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--cal_size", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--partial", type=int, default=0)

    args = parser.parse_args()

    # dataset = TTSDataset(name=args.dataset, percent=args.cal_size)
    cal_set = TTSDataset(name=args.dataset, percent=args.cal_size).dataset
    per_partial = int(len(cal_set)/8)
    cal_set = cal_set[args.partial * per_partial, (args.partial+1) * per_partial]
    inference = LLMInference(args.model)

    # client = AzureOpenAI(
    #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    #     api_key=os.environ["AZURE_OPENAI_API_KEY"],
    #     api_version="2024-02-01"
    # )

    # predictor = conformalTTS(inference, args.alpha, dataset, client=client)
    predictor = conformalParTTS(inference, cal_set, None, args.batch_size)
    # predictor.predict()
    # predictor.budget_force_predict()
    predictor.calibrate()