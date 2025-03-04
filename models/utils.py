from vllm import LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

pretrained_model_dic = {
    "s1-32B": "/mnt/sharedata/hdd/jiaxi/model/s1-32B",
}

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class LLMInference:
    def __init__(self, name):
        if name in pretrained_model_dic:
            self.name = name
            # self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_dic[name], device_map="auto", trust_remote_code=True).eval().half()
            self.model = LLM(pretrained_model_dic[name], tensor_parallel_size=4, gpu_memory_utilization=0.5)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dic[name])
            self.template = self.prompt_template()
        else:
            print("Error: Model Type")
    
    def prompt_template(self):
        if self.name == "s1-32B":
            return lambda p: "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
        