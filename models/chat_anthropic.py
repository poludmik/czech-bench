from langchain_anthropic import ChatAnthropic

def get_llm(model_id="claude-3-haiku-20240307", temperature=0, max_tokens=512):
    llm = ChatAnthropic(model=model_id, temperature=temperature, max_tokens=max_tokens)
    return llm