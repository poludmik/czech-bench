# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Answer the given question about {topic} by choosing one of the four proposed answers.
Do not repeat the chosen answer. Always answer only with the digit corresponding to the chosen answer without any further comment.

"""

few_shot = """Here are some example questions:

{shots}
"""

request = """Answer the following question:
Question:
{question}
Choices:
1) {options[0]}
2) {options[1]}
3) {options[2]}
4) {options[3]}
Answer:
"""

input_variables=["topic", "shots", "question", "options"]
input_types={"topic": "str", "shots": "str", "question": "str", "options": "List[str]"}


prompt_template = task + few_shot + request
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=input_variables, input_types=input_types
)

system_template = task + few_shot
msg_template = request
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(msg_template),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

