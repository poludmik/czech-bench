from langchain_openai import ChatOpenAI

def get_llm(model_id="gpt-3.5-turbo-1106", temperature=0):
    llm = ChatOpenAI(model=model_id, temperature=temperature, request_timeout=20)
    return llm