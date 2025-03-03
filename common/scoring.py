import math
import torch

def ppl_score(model, tokenizer, text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    ppl = math.exp(loss)
    return ppl

# def ppl_score(model, tokenizer, text):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     inputs = tokenizer.encode(text, return_tensors="pt")
#     inputs = inputs.to(device)
#     with torch.no_grad():
#         outputs = model(inputs)
#     logits = outputs.logits
#     labels = inputs[1:]
#     logits = logits[:, :-1, :]
#     softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
#     loss = 0
#     for i in range(len(labels)):
#         token_prob = softmax_probs[i, labels[i]]
#         loss -= torch.log(token_prob)
#     avg_loss = loss / len(labels)
#     ppl = torch.exp(avg_loss)
#     return ppl.item()