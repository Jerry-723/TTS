from models import *
from datasets import *
from predictor import *
from common import *
import argparse
import os
from openai import AzureOpenAI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cal_set", type=str, choices=["aime", "math500", "gsm8k"], default="math500")
    # parser.add_argument("--pred_set", type=str, choices=["aime", "math500", "gsm8k"], default="gsm8k")
    parser.add_argument("--dataset", type=str, choices=["aime", "math500", "gsm8k"], default="gsm8k")
    parser.add_argument("--model", type=str, choices=["s1-32B", "DeepSeek-R1-Distill-Qwen-1.5B", "s1.1-3B", "s1.1-7B", "s1.1-14B"], default="s1.1-3B")
    parser.add_argument("--cal_size", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--partial", type=int, default=0)
    parser.add_argument("--task", type=str, choices=["fullthinking", "fulltrace", "nodup_ans", "predict"], default="predict")

    args = parser.parse_args()

    # cal_set = TTSDataset(name=args.cal_set, percent=args.cal_size)
    # pred_set = TTSDataset(name=args.pred_set, percent=args.cal_size)
    dataset = TTSDataset(name=args.dataset, percent=args.cal_size)
    inference = LLMInference(args.model)

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01"
    )

    # predictor = conformalTTS(inference, args.alpha, dataset, client=client)
    # predictor = conformalParTTS(inference, cal_set, pred_set, args.batch_size, args.partial, client)
    predictor = EfficientPred(inference, dataset, args.partial, args.batch_size, args.alpha)
    if args.task == "fullthinking":
        predictor.forcefull(10000)
    if args.task == "fulltrace":
        predictor.full_trace()
    if args.task == "nodup_ans":
        predictor.nodup_answer(client)
    if args.task == "predict":
        # predictor.predict_cluster()
        predictor.calibrate_pred()
