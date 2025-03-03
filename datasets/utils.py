import json
import csv
import pandas as pd
import random
random.seed(2025)

class TTSDataset:
    def __init__(self, name, percent):
        self.name = name
        self.percent = percent
        self.dataset = self.load_dataset()
        self.cal_dataset, self.test_dataset = self.cal_test(percent)

    def load_dataset(self):
        dataset = []
        if self.name == "aime":
            df = pd.read_parquet("/data/home/jiaxi/home/TTS/datasets/assets/aime_2024_problems.parquet")
            for i, row in df.iterrows():
                dataset.append({
                    "question": row["Problem"],
                    "solution": row["Solution"],
                    "answer": row["Answer"]
                })

        elif self.name == "gpqa":
            # with open("/data/home/jiaxi/home/TTS/datasets/gpqa_diamond.csv", "r") as f:
            #     reader = csv.DictReader(f)
            #     for row in reader:
            pass

        elif self.name == "math500":
            with open("/data/home/jiaxi/home/TTS/datasets/assets/math500.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line)
                    dataset.append({
                        "question": data["problem"],
                        "solution": data["solution"],
                        "answer": data["answer"]
                    })

        return dataset
    
    def cal_test(self, percent):
        random.shuffle(self.dataset)
        cal_size = int(len(self.dataset) * percent)
        cal_dataset = self.dataset[:cal_size]
        test_dataset = self.dataset[cal_size:]
        return cal_dataset, test_dataset
