from datasets import load_dataset, load_from_disk
from langchain.chains.prompt_selector import is_chat_model
from langchain_core.output_parsers.string import StrOutputParser
import evaluate
import json
import os
import numpy as np
import time
from datetime import datetime
from .prompts import PROMPT_SELECTOR

local_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(local_dir))

BENCHMARK = "Klokanek"

class Evaluator:
    def __init__(self, local=False):
        print(f"\nInitializing {BENCHMARK} evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        self.dataset = load_dataset("ctu-aic/hynky/klokan-qa", split="train")
        #self.dataset.save_to_disk(local_dir + "/data/test")

    def load_local(self):
        print("Loading dataset locally")
        self.dataset = load_dataset("parquet", data_files={'train': local_dir + "/data/train/train-00000-of-00001.parquet"}, split="train")
    
    def run_eval(self, llm, result_file, stop_idx=np.inf):
        info = f'\nCommencing {BENCHMARK} evaluation at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}'
        with open (result_file, "a") as rf:
            rf.write(f"\n\n*** {BENCHMARK} ***" + info + "\n")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        correct = [0]*6
        total = [0]*6
        correct_all = 0
        parse_fails = 0
        count = 0
        cum_time = 0.

        for i, example in enumerate(self.dataset):
            if i+1 > stop_idx:
                break
            print(f"\rExample {i+1} / {len(self.dataset)}", end="")
            question = example["question"]
            A = example["answers.A"]
            B = example["answers.B"]
            C = example["answers.C"]
            D = example["answers.D"]
            E = example["answers.E"]
            cat = example["category"]
            gt = example["correct_answer"]

            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(question=question, optionA=A, optionB=B, optionC=C, optionD=D, optionE=E).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(question=question, optionA=A, optionB=B, optionC=C, optionD=D, optionE=E).text)    
                result = str_parser.invoke(result)
                end_time = time.time()
                print(result)
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue
            
            if result == gt:
                correct[cat] += 1
                correct_all += 1
            elif result not in "ABCDE":
                parse_fails += 1
                continue
            total[cat] += 1
            count += 1
            cum_time += end_time - start_time
            
        print("\nComputing metrics")

        lines = "\nResults:\n"
        if count > 0:
            for i in range(6):
                if total[i] > 0:
                    lines += f"Category {i} accuracy: {correct[i]/total[i]*100:.2f}% ({correct[i]}/{total[i]})\n"
                else:
                    lines += f"Category {i} accuracy: N/A\n"
            lines += f"Overall accuracy: {correct_all/count*100:.2f}% ({correct_all}/{count})\n"
            lines += f"Average inference time: {cum_time/count:.2f}s\n"
        
        lines += f"Total valid examples used: {count}\n"
        lines += f"Unparseable answers: {parse_fails}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print(lines)
