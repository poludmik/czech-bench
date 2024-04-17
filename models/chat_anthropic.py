from langchain_anthropic import ChatAnthropic

def get_llm(model_id="claude-3-haiku-20240307", temperature=0):
    llm = ChatAnthropic(model=model_id, temperature=temperature)
    return llm