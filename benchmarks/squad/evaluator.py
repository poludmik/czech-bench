#from datasets import load_dataset
from langchain.chains.prompt_selector import is_chat_model
from langchain_core.output_parsers.string import StrOutputParser
from .squad_v2 import SquadV2
import evaluate
import json
import sys
import os
import numpy as np
import time
import copy
from datetime import datetime
from .prompts import PROMPT_SELECTOR

local_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(local_dir))
sys.path.append(home_dir + "/tools/morphology")
from lemmatization import MorphoDiTa
from word_roots import DeriNet


class Evaluator:
    def __init__(self, local=False):
        print("\nInitializing SQuAD evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()
        
        print("Loading morphological tools")
        self.lemmatizer = MorphoDiTa()
        if not self.lemmatizer.lemmatizer:
            self.lemmatizer = False
        self.root_lexicon = None
        if not self.lemmatizer:
            print("Failed to load lemmatizer model, morphological analysis will be skipped")
        else:
            try:
                self.root_lexicon = DeriNet()
            except:
                print("Failed to load root lexicon, root extraction will be skipped")

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        raise NotImplementedError  # This dataset is not available on Huggin Face ATM

    def load_local(self):      
        print("Loading dataset locally")
        self.dataset = list(SquadV2()._generate_examples(local_dir + "/data/dev-v2.0.json"))
        self.dataset = self.dataset[:4000] # To save evaluation costs
    
    def morpho_analyze(self, answer):
        tokens = [tok.rstrip(",.?!") for tok in answer.lower().rstrip('\r\n').split()]
        lemmas = ""
        roots = ""
        for token in tokens:
            lemma = self.lemmatizer.lemmatize(token)
            lemmas += lemma + " "      
            if self.root_lexicon is not None:
                roots += self.root_lexicon.get_root(lemma) + " "
        return lemmas, roots
    
    def run_eval(self, llm, result_file, stop_idx=np.inf):
        info = f'\nCommencing SQuAD evaluation at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}'
        print(info)
        with open (result_file, "a") as rf:
            rf.write("\n\n*** SQuAD ***" + info + "\n")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        references = []
        predictions = []
        ref_lemmas = []
        pred_lemmas = []
        ref_roots = []
        pred_roots = []
        na_gt = []
        na_pr = []
        parse_fails = 0
        count = 0
        cum_time = 0.

        for i, example in enumerate(self.dataset):
            if i+1 > stop_idx:
                break
            print(f"\rExample {i+1} / {len(self.dataset)}", end="")
            context = example["context"]
            question = example["question"]

            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(context=context, question=question).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(context=context, question=question).text)
                result = str_parser.invoke(result)
                end_time = time.time()
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue
            
            result = result.split("\n")[0]
            #print(result)
            if result.strip() == "-":
                a_text = ""
                a_prob = 1
            else:
                a_text = result
                a_prob = 0
            no_answer_gt = 0 if example["answers"]["answer_start"] else 1
            na_gt.append(no_answer_gt)
            na_pr.append(a_prob)

            id = example["id"]
            ref = {
                "answers": example["answers"],
                "id": id,
            }
            ans = {
                "prediction_text": a_text,
                "id": id,
                "no_answer_probability": a_prob
            }
            references.append(ref)
            predictions.append(ans)

            if self.lemmatizer:
                ans_lemmas = []
                ans_roots = []
                for ans_text in example["answers"]["text"]:
                    lems, roots = self.morpho_analyze(ans_text)
                    ans_lemmas.append(lems)
                    ans_roots.append(roots)

                ref_lem_dict = copy.deepcopy(ref)
                ref_lem_dict["answers"]["text"] = ans_lemmas
                ref_root_dict = copy.deepcopy(ref)
                ref_root_dict["answers"]["text"] = ans_roots

                lems, roots = self.morpho_analyze(a_text)
                pred_lem_dict = copy.deepcopy(ans)
                pred_lem_dict["prediction_text"] = lems
                pred_root_dict = copy.deepcopy(ans)
                pred_root_dict["prediction_text"] = roots

                ref_lemmas.append(ref_lem_dict)
                pred_lemmas.append(pred_lem_dict)
                if self.root_lexicon is not None:
                    ref_roots.append(ref_root_dict)
                    pred_roots.append(pred_root_dict)
            count += 1
            cum_time += end_time - start_time

        print("\nComputing metrics")

        lines = "\nResults:\n"
        if count > 0:
            metric = evaluate.load("squad_v2")
            res = metric.compute(predictions=predictions, references=references)
            lines += f"Exact Match: {res['exact']:.2f}\n"
            lines += f"BoW F1: {res['f1']:.2f}\n"

            if self.lemmatizer:
                res_lemma = metric.compute(predictions=pred_lemmas, references=ref_lemmas)
                lines += f"Lemmas Exact Match: {res_lemma['exact']:.2f}\n"
                lines += f"Lemmas BoW F1: {res_lemma['f1']:.2f}\n"

                if self.root_lexicon is not None:
                    res_root = metric.compute(predictions=pred_roots, references=ref_roots)
                    lines += f"Roots Exact Match: {res_root['exact']:.2f}\n"
                    lines += f"Roots BoW F1: {res_root['f1']:.2f}\n"
            
            metric = evaluate.load("accuracy")
            res_cls = metric.compute(predictions=na_pr, references=na_gt)
            lines += f"No-Answer-Detection Accuracy: {res_cls['accuracy']*100:.2f}\n"

            metric = evaluate.load("f1")
            res_cls = metric.compute(predictions=na_pr, references=na_gt, average='macro')
            lines += f"No-Answer-Detection F1: {res_cls['f1']*100:.2f}\n"

            lines += f"Average inference time: {cum_time/count:.2f}s\n"
        
        lines += f"Total valid examples used: {count}\n"
        lines += f"Unparseable answers: {parse_fails}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print(lines)
