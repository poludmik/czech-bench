import argparse
import yaml
import os
from datetime import datetime
from types import SimpleNamespace
import logging
import traceback
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", default="eval_config.yml", help="Path to custom config file")
logging.disable(logging.WARNING)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Load config
    with open(args.config_file, "r") as cf:
        try:
            config = yaml.safe_load(cf)
            cfg = SimpleNamespace(**config)
        except yaml.YAMLError:
            print(traceback.format_exc())
            raise Exception("Config file could not be parsed")
    
    # Create LLM
    try:
        exec(f'from models.{cfg.model_name} import get_llm')
        llm = get_llm()
    except Exception:
        print(traceback.format_exc())
        raise Exception("Model could not be imported")

    # Run benchmarks  
    result_file = os.path.abspath(f'./results/{cfg.model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

    for bench in cfg.benchmarks:
        try:
            if not bench["use"]:
                continue
            exec(f"from benchmarks.{bench['name']}.evaluator import Evaluator")
            evaluator = Evaluator(bench['local'])
            stop_idx = cfg.stop_idx
            if stop_idx is None:
                stop_idx = np.inf
            evaluator.run_eval(llm, result_file, stop_idx)
        except Exception:
            print(traceback.format_exc())
            info = f"{bench['name']} evaluation skipped due to uncatched exception"
            with open(result_file, "a") as rf:
                rf.write(info + "\n")
            print(info)

    info = f'\nEvaluation finished at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}\n'
    with open(result_file, "a") as rf:
        rf.write("\n" + info + "\n")
    print(info + f'Results can be found at {result_file}')