from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from ollama import Client, Options

class CustomOllama(LLM):
    model : str = "llama2"
    base_url : str = "http://localhost:11434"
    temperature : Optional[float] = None
    ollama : Client = None

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", temperature: Optional[float] = None):
        super().__init__()
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        self.ollama = Client(host=base_url, timeout=20)

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}
    
    def invoke(self, prompt: str) -> str:
        return self.ollama.generate(model=self.model, prompt=prompt, raw=True, options=Options(temperature=self.temperature))["response"]
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.invoke(prompt)


def get_llm(model_id="llama2", url="http://localhost:11434", temperature=None):
    llm = CustomOllama(model=model_id, base_url=url, temperature=temperature)
    return llm