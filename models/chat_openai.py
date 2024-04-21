from langchain_openai import ChatOpenAI

def get_llm(model_id="gpt-3.5-turbo-0125", temperature=0, max_tokens=512):
    llm = ChatOpenAI(model=model_id, temperature=temperature, max_tokens=max_tokens, request_timeout=20)
    return llm