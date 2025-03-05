import math
import torch
from vllm import SamplingParams

## transformers version
# def ppl_score(model, tokenizer, text):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     inputs = tokenizer(text, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss
#     ppl = math.exp(loss)
#     return ppl

## VLLM version
def ppl_score(model, text):
    sampling_params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1
    )
    o = model.generate(
        text,
        sampling_params=sampling_params
    )
    logprob_total = 0
    for i in o[0].prompt_logprobs:
        if i is not None:
            _, logprob = next(iter(i.items()))
            logprob_total += logprob.logprob 
    avg_logprob = logprob_total / (len(o[0].prompt_logprobs)-1)
    ppl = math.exp(-avg_logprob)
    return ppl