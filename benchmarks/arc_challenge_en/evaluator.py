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

BENCHMARK = "ARC Challenge EN"

class Evaluator:
    def __init__(self, local=False):
        print(f"\nInitializing {BENCHMARK} evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        self.dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        #self.dataset.save_to_disk(local_dir + "/data/test")

    def load_local(self):
        print("Loading dataset locally")
        self.dataset = load_from_disk(local_dir + "/data/test")
    
    def run_eval(self, llm, result_file, stop_idx=np.inf):
        info = f'\nCommencing {BENCHMARK} evaluation at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}'
        with open (result_file, "a") as rf:
            rf.write(f"\n\n*** {BENCHMARK} ***" + info + "\n")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        correct = 0
        parse_fails = 0
        count = 0
        cum_time = 0.

        for i, example in enumerate(self.dataset):
            if i+1 > stop_idx:
                break
            print(f"\rExample {i+1} / {len(self.dataset)}", end="")

            if len(example["choices"]["text"]) != len(example["choices"]["label"]):
                continue
            question = example["question"]
            choice_selection = ""
            for i in range(len(example["choices"]["text"])):
                choice_selection += f"{example['choices']['label'][i]}) {example['choices']['text'][i]}\n"
            gt = example["answerKey"]

            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(question=question, choices=choice_selection).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(question=question, choices=choice_selection).text)    
                result = str_parser.invoke(result)
                end_time = time.time()
                res = result.split()[0].strip().strip(")")
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue

            if res == gt:
                correct += 1
            elif len(res) > 1 or res not in "ABCD":
                parse_fails += 1
                continue
            count += 1
            cum_time += end_time - start_time
            
        print("\nComputing metrics")

        lines = "\nResults:\n"
        if count > 0:
            lines += f"Accuracy: {correct/count*100:.2f}\n"
            lines += f"Average inference time: {cum_time/count:.2f}s\n"
        
        lines += f"Total valid examples used: {count}\n"
        lines += f"Unparseable answers: {parse_fails}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print(lines)
