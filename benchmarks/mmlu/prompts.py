# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Odpověz na zadanou otázku na téma {topic} výběrem jedné ze čtyř navržených odpovědí.
Zvolenou odpověď neopakuj. Vždy odpovídej pouze číslem odpovídajícím vybrané odpovědi bez dalšího komentáře.

"""

few_shot = """Zde je pět ukázkových příkladů:

{shots}
"""

request = """Odpověz na následující otázku:
Otázka:
{question}
Možnosti:
1) {options[0]}
2) {options[1]}
3) {options[2]}
4) {options[3]}
Odpověď:
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

