from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI

def get_llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", request_timeout=20)
    return llm