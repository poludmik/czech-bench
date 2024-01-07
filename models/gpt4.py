#from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

def get_llm():
    llm = OpenAI(model="gpt-4-1106-preview", request_timeout=20)
    return llm