from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, QuantoConfig
import torch


def get_llm(model_id="google/flan-t5-xl", causal=True, do_sample=False, max_new_tokens=512, precision="auto", device_map="auto", **kwargs):

    if precision is not None:
        if precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif precision == "fp32":
            kwargs["torch_dtype"] = torch.float32
        elif precision == "auto":
            kwargs["torch_dtype"] = 'auto'
        elif precision == "fp8":
            kwargs["quantization_config"] = QuantoConfig(weights="float8")
        elif precision in ["int8", "int4", "int2"]:
            kwargs["quantization_config"] = QuantoConfig(weights=precision)
        else:
            print(f"Unknown precision '{precision}', setting to 'auto'")
            kwargs["torch_dtype"] = 'auto'
    kwargs['do_sample'] = do_sample

    pipe_kwargs = {}
    if causal:
        pipe_kwargs['return_full_text'] = False

    pipe = pipeline(model=model_id, device_map=device_map, max_new_tokens=max_new_tokens, model_kwargs=kwargs, **pipe_kwargs)

    hf = HuggingFacePipeline(pipeline=pipe)
    print(f"Model loaded on {pipe.model.device}")
    
    return hf
