from langchain.chains.prompt_selector import is_chat_model
from langchain_core.output_parsers.string import StrOutputParser
from datasets import load_dataset, load_from_disk
import os
import numpy as np
import time
import re
from datetime import datetime
from .prompts import PROMPT_SELECTOR

local_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(local_dir))

BENCHMARK = "GSM8K EN"

class Evaluator:
    def __init__(self, local=False):
        print(f"\nInitializing {BENCHMARK} evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        self.dataset = load_dataset("gsm8k", "main", split="test")

        def map_fn(example):
            match = re.search(r'^(.*)####', example['answer'], re.DOTALL)
            if match:
                example['thoughts'] = re.sub(r'<<.*?>>', '', match.group(1))
            else:
                example['thoughts'] = ""
            match = re.search(r'#### ([\d,-]+)$', example['answer'])
            ans = match.group(1)
            ans = ans.replace(',', '')
            example['answer'] = int(ans)
            return example
        
        self.dataset = self.dataset.map(map_fn)
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
        rel_err_count = 0
        abs_err_sum = 0
        rel_err_sum = 0
        cum_time = 0.

        for i, example in enumerate(self.dataset):
            if i+1 > stop_idx:
                break
            print(f"\rExample {i+1} / {len(self.dataset)}", end="")
            question = example["question"]
            gt = example["answer"]

            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(question=question).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(question=question).text)
                result = str_parser.invoke(result)
                end_time = time.time()
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue

            try:
                match = re.search(r'#### ([\d,-]+)', result)
                ans = match.group(1)
                ans = ans.replace(',', '')
                prediction = int(ans)
                if prediction == gt:
                    correct += 1
                abs_error = abs((gt - prediction))
                abs_err_sum += abs_error
                if gt != 0:
                    rel_error = abs(abs_error / gt)
                    rel_err_sum += rel_error
                    rel_err_count += 1
            except:
                #print(result)
                parse_fails += 1
                continue
            count += 1
            cum_time += end_time - start_time
            
        print("\nComputing metrics")
        
        lines = "\nResults:\n"
        if count > 0:
            lines += f"Final result accuracy: {correct/count*100:.2f}\n"
            lines += f"Mean absolute error: {abs_err_sum/count:.2f}\n"
            if rel_err_count > 0:
                lines += f"Mean relative error: {rel_err_sum/rel_err_count:.2f}\n"
            lines += f"Average inference time: {cum_time/count:.2f}s\n"
        
        lines += f"Total valid examples used: {count}\n"
        lines += f"Unparseable answers: {parse_fails}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print(lines)
