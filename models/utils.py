from vllm import LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

pretrained_model_dic = {
    "s1-32B": "/mnt/sharedata/hdd/jiaxi/model/s1-32B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "/mnt/sharedata/hdd/jiaxi/model/DeepSeek-R1-Distill-Qwen-1.5B",
    "s1.1-3B": "/mnt/sharedata/hdd/zhanghx/ssd2/zhanghx/models/s1.1-3B",
    "s1.1-7B": "/mnt/sharedata/hdd/zhanghx/ssd2/zhanghx/models/s1.1-7B",
    "s1.1-14B": "/mnt/sharedata/hdd/zhanghx/ssd2/zhanghx/models/s1.1-14B",
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
            self.model = LLM(pretrained_model_dic[name],
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.9,
                            # max_num_batched_tokens=512,
                            enable_chunked_prefill=True)
            # self.model = None
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dic[name])
            self.template = self.prompt_template()
        else:
            print("Error: Model Type")
    
    def prompt_template(self):
        if self.name == "s1-32B" or self.name == "s1.1-3B" or self.name == "s1.1-7B" or self.name == "s1.1-14B":
            return lambda p: "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
        elif self.name == "DeepSeek-R1-Distill-Qwen-1.5B":
            return lambda p: self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": p}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
        