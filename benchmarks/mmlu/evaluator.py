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
from .topics import topics, topic_translations

local_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(local_dir))

BENCHMARK = "MMLU"

class Evaluator:
    def __init__(self, local=False):
        print(f"\nInitializing {BENCHMARK} evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        raise NotImplementedError("Data not available on Hugging Face. Please use the local option.")

    def load_local(self):
        print("Loading dataset locally")
        test = load_from_disk(local_dir + "/data/test")
        dev = load_from_disk(local_dir + "/data/dev")
        self.dataset = {"test": test, "dev": dev}

    def run_eval(self, llm, result_file, stop_idx=np.inf):
        info = f'\nCommencing {BENCHMARK} evaluation at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}'
        with open (result_file, "a") as rf:
            rf.write(f"\n\n*** {BENCHMARK} ***" + info + "\n")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        parse_fails = 0
        count = 0
        cum_time = 0.
        all_accuracies = []

        with open(result_file, "a") as rf:
            rf.write("\nInterim Results:\n\n")

        for category in topics.keys():
            cat_accuracies = []
            for topic in topics[category]:
                dev_set = self.dataset['dev'].filter(lambda example: example['subject'] == topic)
                test_set = self.dataset['test'].filter(lambda example: example['subject'] == topic)

                shots = ""
                for i in range(5):
                    shots += f"Otázka:\n{dev_set[i]['question']}\nMožnosti:\n"
                    for j in range(4):
                        shots += f"{j+1}) {dev_set[i]['choices'][j]}\n"
                    shots += f"Odpověď:\n{dev_set[i]['answer']}\n\n"

                correct = 0
                topic_count = 0
                for i, example in enumerate(test_set):
                    if i+1 > stop_idx:
                        break
                    print(f"\rCategory: {category}, Topic: {topic}, Example: {i+1} / {len(test_set)}", end="")
                    if len(example["choices"]) != 4:
                        continue
                    question = example["question"]
                    choices = example["choices"]
                    gt = example["answer"] + 1

                    try:
                        start_time = time.time()
                        if is_chat_model(llm):
                            result = llm.invoke(prompt.format_prompt(topic=topic_translations[topic], shots=shots, question=question, options=choices).to_messages())
                        else:
                            result = llm.invoke(prompt.format_prompt(topic=topic_translations[topic], shots=shots, question=question, options=choices).text)    
                        result = str_parser.invoke(result)
                        end_time = time.time()
                    except Exception as e:
                        print(f"\nExample skipped due to an LLM Error: {e}")
                        continue
                    
                    try:
                        prediction = int(result)
                    except:
                        #print(result)
                        parse_fails += 1
                        continue
                    if prediction == gt:
                        correct += 1
                    count += 1
                    topic_count += 1
                    cum_time += end_time - start_time

                acc = float("nan")
                if correct > 0:
                    acc = correct / topic_count
                    cat_accuracies.append(acc)
                    all_accuracies.append(acc)
                result = f"{topic} accuracy: {acc*100:.2f}"
                with open(result_file, "a") as rf:
                    rf.write(result + "\n")
                print("\n" + result)

            acc = float("nan")
            if len(cat_accuracies) > 0:
                acc = np.mean(cat_accuracies)
            result = f"{category} average accuracy: {acc*100:.2f}"
            with open(result_file, "a") as rf:
                rf.write(result + "\n\n")
            print(result)

        lines = "Final results:\n"
        acc = float("nan")
        if len(all_accuracies) > 0:
            acc = np.mean(all_accuracies)
        lines += f"Total average accuracy: {acc*100:.2f}\n"     
        lines += f"Total valid examples used: {count}\n"
        lines += f"Unparseable answers: {parse_fails}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print("\n" + lines)
