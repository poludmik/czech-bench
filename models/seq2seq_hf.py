from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from torch import cuda
import torch


def get_llm(model_id="CohereForAI/aya-101", do_sample=False, max_new_tokens=512, precision="auto", **kwargs):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
    else:
        if precision != "auto":
            print(f"Precision '{precision}' not recognized, using 'auto' precision")
        kwargs["torch_dtype"] = 'auto'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', do_sample=do_sample, **kwargs)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
    hf = HuggingFacePipeline(pipeline=pipe)
    print(f"Model loaded on {device}")
    
    return hf
