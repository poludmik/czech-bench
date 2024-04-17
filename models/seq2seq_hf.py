from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from torch import cuda
import torch


def get_llm(model_id="CohereForAI/aya-101", precision="full", load_in_8bit=False):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if precision == "half":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', do_sample=False, load_in_8bit=load_in_8bit, torch_dtype=torch.float16)
    else:
        if precision != "full":
            print(f"Precision {precision} not recognized. Defaulting to full precision.")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', do_sample=False, load_in_8bit=load_in_8bit)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    hf = HuggingFacePipeline(pipeline=pipe)
    print(f"Model loaded on {device}")
    
    return hf
