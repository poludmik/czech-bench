from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import pipeline, QuantoConfig
import torch


class CustomChatLlama3(BaseChatModel):
    pipeline: Any = None
    dosample: bool = False

    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", do_sample=False, max_new_tokens=512, precision="auto", device_map="auto", **kwargs):
        super().__init__()
        
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

        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            max_new_tokens=max_new_tokens,
            model_kwargs=kwargs,
            device_map=device_map,
        )


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        system = messages[0].content
        human = messages[1].content

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": human},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=self.dosample,
        )
        
        result = outputs[0]["generated_text"][len(prompt):]

        message = AIMessage(content=result)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "custom-llama3"
    

def get_llm(model_id="meta-llama/Meta-Llama-3-8B-Instruct", do_sample=False, max_new_tokens=512, precision="auto", device_map="auto", **kwargs):
    llm = CustomChatLlama3(
        model_id=model_id,
        do_sample=do_sample, 
        max_new_tokens=max_new_tokens, 
        precision=precision, 
        device_map=device_map, 
        **kwargs
    )
    return llm
