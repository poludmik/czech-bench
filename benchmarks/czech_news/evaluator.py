from datasets import load_dataset, load_from_disk, Dataset
from langchain.chains.prompt_selector import is_chat_model
from langchain_core.output_parsers.string import StrOutputParser
import evaluate
import json
import os
import numpy as np
import time
from datetime import datetime
import pandas as pd
from .prompts import PROMPT_SELECTOR

local_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(local_dir))


class Evaluator:
    def __init__(self, local=False):
        print("\nInitializing Czech News evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()

    def filter_data(self, data, nc = 200, rs = 42):
        data = data.filter(lambda example: example["category"] in [1, 2, 3, 4, 7])
        df = pd.DataFrame(data)
        g = df.groupby('category')
        df = g.apply(lambda x: x.sample(nc, random_state=rs).reset_index(drop=True))
        df['category'].replace(7, 5, inplace=True)
        data = Dataset.from_pandas(df).shuffle(seed=rs)
        return data

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        data = load_dataset("hynky/czech_news_dataset_v2", split="test")
        self.dataset = self.filter_data(data)
        #self.dataset.save_to_disk(local_dir + "/data/test")

    def load_local(self):
        print("Loading dataset locally")
        self.dataset = load_from_disk(local_dir + "/data/test")
    
    def run_eval(self, llm, result_file, stop_idx=np.inf):
        info = f'\nCommencing Czech News evaluation at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}'
        print(info)
        with open (result_file, "a") as rf:
            rf.write("\n\n*** Czech News ***" + info + "\n")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        labels = []
        predictions = []
        parse_fails = 0
        count = 0
        cum_time = 0.

        for i, example in enumerate(self.dataset):
            if i+1 > stop_idx:
                break
            print(f"\rExample {i+1} / {len(self.dataset)}", end="")
            brief = example["brief"]
            #content = example["content"]
            label = example["category"]
            
            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(brief=brief).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(brief=brief).text)
                result = str_parser.invoke(result)
                end_time = time.time()
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue
            
            try:
                prediction = int(result)
                labels.append(label)
                predictions.append(prediction)
            except:
                parse_fails += 1
                continue
            count += 1
            cum_time += end_time - start_time
            
        print("\nComputing metrics")

        lines = "\nResults:\n"
        if count > 0:
            metric = evaluate.load("accuracy")
            res = metric.compute(predictions=predictions, references=labels)
            lines += f"Accuracy: {res['accuracy']*100:.2f}\n"

            metric = evaluate.load("f1")
            res_f1 = metric.compute(predictions=predictions, references=labels, average='weighted')
            lines += f"F1: {res_f1['f1']*100:.2f}\n"

            lines += f"Average inference time: {cum_time/count:.2f}s\n"

        lines += f"Total valid examples used: {count}\n"
        lines += f"Unparseable answers: {parse_fails}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print(lines)
