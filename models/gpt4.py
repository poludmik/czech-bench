from langchain.chat_models import ChatOpenAI

def get_llm():
    llm = ChatOpenAI(model="gpt-4-1106-preview", request_timeout=20)
    return llm