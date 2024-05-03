from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
import torch


def get_llm(model_id="google/flan-t5-xl", task=None, do_sample=False, max_new_tokens=512, precision="auto", device_map="auto", **kwargs):

    if precision is not None:
        if precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif precision == "fp32":
            kwargs["torch_dtype"] = torch.float32
        elif precision == "auto":
            kwargs["torch_dtype"] = 'auto'
        else:
            print(f"Unknown precision '{precision}', setting to 'auto'")
            kwargs["torch_dtype"] = 'auto'
    kwargs['do_sample'] = do_sample

    if task is None:
        pipe = pipeline(model=model_id, device_map=device_map, max_new_tokens=max_new_tokens, model_kwargs=kwargs)
    else:
        pipe = pipeline(task, model=model_id, device_map=device_map, max_new_tokens=max_new_tokens, model_kwargs=kwargs)

    hf = HuggingFacePipeline(pipeline=pipe)
    print(f"Model loaded on {pipe.model.device}")
    
    return hf
