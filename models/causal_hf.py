from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch import cuda
import torch


def get_llm(model_id='bigscience/bloomz-7b1-mt', precision="full", **kwargs):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if precision == "half":
        kwargs["torch_dtype"] = torch.float16
    else:
        if precision != "full":
            print(f"Precision {precision} not recognized, using full precision")
        kwargs["torch_dtype"] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', do_sample=False, **kwargs)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, return_full_text=False)
    hf = HuggingFacePipeline(pipeline=pipe)
    print(f"Model loaded on {device}")
    
    return hf
