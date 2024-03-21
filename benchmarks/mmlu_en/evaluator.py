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

BENCHMARK = "MMLU EN"

topics = {
            "Humanities": [
                "formal_logic",
                "high_school_european_history",
                "high_school_us_history",
                "high_school_world_history",
                "international_law",
                "jurisprudence",
                "logical_fallacies",
                "moral_disputes",
                "moral_scenarios",
                "philosophy",
                "prehistory",
                "professional_law",
                "world_religions",
            ],
            "Social Sciences": [
                "econometrics",
                "high_school_geography",
                "high_school_government_and_politics",
                "high_school_macroeconomics",
                "high_school_microeconomics",
                "high_school_psychology",
                "human_sexuality",
                "professional_psychology",
                "public_relations",
                "security_studies",
                "sociology",
                "us_foreign_policy",
            ],
            "STEM": [
                "abstract_algebra",
                "anatomy",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
                "machine_learning",
            ],
            "Other": [
                "business_ethics",
                "clinical_knowledge",
                "college_medicine",
                "global_facts",
                "human_aging",
                "management",
                "marketing",
                "medical_genetics",
                "miscellaneous",
                "nutrition",
                "professional_accounting",
                "professional_medicine",
                "virology",
            ]
        }

class Evaluator:
    def __init__(self, local=False):
        print(f"\nInitializing {BENCHMARK} evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        self.dataset = load_dataset("cais/mmlu", 'all')

    def load_local(self):
        print("Loading dataset locally")
        raise NotImplementedError("Local data not available")
    
    def run_eval(self, llm, result_file, stop_idx=np.inf):
        info = f'\nCommencing {BENCHMARK} evaluation at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}'
        with open (result_file, "a") as rf:
            rf.write(f"\n\n*** {BENCHMARK} ***" + info + "\n")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        parse_fails = 0
        count = 0
        cum_time = 0.
        lines = "\nResults:\n\n"
        all_accuracies = []

        for category in topics.keys():
            cat_accuracies = []
            for topic in topics[category]:
                dev_set = self.dataset['dev'].filter(lambda example: example['subject'] == topic)
                test_set = self.dataset['test'].filter(lambda example: example['subject'] == topic)

                shots = ""
                for i in range(5):
                    shots += f"Question:\n{dev_set[i]['question']}\nChoices:\n"
                    for j in range(4):
                        shots += f"{j+1}) {dev_set[i]['choices'][j]}\n"
                    shots += f"Answer:\n{dev_set[i]['answer']}\n\n"

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
                            result = llm.invoke(prompt.format_prompt(shots=shots, question=question, options=choices).to_messages())
                        else:
                            result = llm.invoke(prompt.format_prompt(shots=shots, question=question, options=choices).text)    
                        result = str_parser.invoke(result)
                        end_time = time.time()
                    except Exception as e:
                        print(f"\nExample skipped due to an LLM Error: {e}")
                        continue
                    
                    try:
                        prediction = int(result)
                    except:
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
                print("\n" + result)
                lines += result + "\n"

            acc = float("nan")
            if len(cat_accuracies) > 0:
                acc = np.mean(cat_accuracies)
            result = f"{category} average accuracy: {acc*100:.2f}"
            print(result)
            lines += result + "\n\n"

        acc = float("nan")
        if len(all_accuracies) > 0:
            acc = np.mean(all_accuracies)
        result = f"Total average accuracy: {acc*100:.2f}"
        #print(result)
        lines += result + "\n\n"
        
        lines += f"Total valid examples used: {count}\n"
        lines += f"Unparseable answers: {parse_fails}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print(lines)
