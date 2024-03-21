# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt_template = """Answer the given question by choosing one of the four proposed answers.
Always answer only by the digit corresponding to the chosen answer without any further comment.

Here are 5 example questions:

{shots}
Answer the following question:
Question:
{question}
Choices:
1) {options[0]}
2) {options[1]}
3) {options[2]}
4) {options[3]}
Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["shots", "question", "options"], input_types={"shots": "str", "question": "str", "options": "List[str]"}
)

system_template = """Answer the given question by choosing one of the four proposed answers.
Always answer only by the digit corresponding to the chosen answer without any further comment.

Here are 5 example questions:

{shots}
"""

msg_template = """Answer the following question:
Question:
{question}
Choices:
1) {options[0]}
2) {options[1]}
3) {options[2]}
4) {options[3]}
Answer:
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(msg_template),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

