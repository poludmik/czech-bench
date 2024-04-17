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

BENCHMARK = "Belebele"

class Evaluator:
    def __init__(self, local=False):
        print(f"\nInitializing {BENCHMARK} evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        self.dataset = load_dataset("facebook/belebele", split="ces_Latn")
        self.dataset.save_to_disk(local_dir + "/data/test")
        example_idcs = [0, 3, 5, 4, 8]
        self.dataset = self.dataset.select(
            (
                i for i in range(len(self.dataset)) 
                if i not in set(example_idcs)
            )
        )

    def load_local(self):
        print("Loading dataset locally")
        dataset = load_from_disk(local_dir + "/data/test")
        example_idcs = [0, 3, 5, 4, 8]
        self.dataset = dataset.select(
            (
                i for i in range(len(dataset)) 
                if i not in set(example_idcs)
            )
        )
    
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
            
            context = example['flores_passage']
            question = example['question']
            a1 = example['mc_answer1']
            a2 = example['mc_answer2']
            a3 = example['mc_answer3']
            a4 = example['mc_answer4']
            gt = int(example['correct_answer_num'])
            
            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(
                        context=context, question=question, option1=a1, option2=a2, option3=a3, option4=a4).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(
                        context=context, question=question, option1=a1, option2=a2, option3=a3, option4=a4).text)    
                result = str_parser.invoke(result)
                end_time = time.time()
                res = result.split()[0].strip().strip(")")
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue
            
            try:
                prediction = int(res)
            except:
                #print(f"\n{result}\n")
                parse_fails += 1
                continue
            if prediction == gt:
                correct += 1
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
