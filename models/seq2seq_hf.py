from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from torch import cuda
import torch


def get_llm(model_id="CohereForAI/aya-101", device_map='auto', do_sample=False, max_new_tokens=512, precision="full", **kwargs):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if precision == "half":
        kwargs["torch_dtype"] = torch.float16
    else:
        if precision != "full":
            print(f"Precision {precision} not recognized, using full precision")
        kwargs["torch_dtype"] = 'auto'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device_map, do_sample=do_sample, **kwargs)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
    hf = HuggingFacePipeline(pipeline=pipe)
    print(f"Model loaded on {device}")
    
    return hf
