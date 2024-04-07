# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """V zadané větě je jedno slovo nahrazeno výrazem '____'. Z nabídnutých možností vyber slovo, po jehož dolnění bude zadaná věta gramaticky správná.
Zvolenou odpověď neopakuj. Odpovídej vždy pouze číslicí odpovídajícím zvolené odpovědi bez dašího komentáře.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Věta:
Moje zamyšlení, jak jste si ____, je hlavně o mrhání.
Možnosti:
1) všimla
2) všimlo
3) všimli
4) všimly
5) všiml
Odpověď:
3

Věta:
Jeho kapacita ____ až 36 tisíc diváků.
Možnosti:
1) byla
2) bylo
3) byli
4) byly
5) byl
Odpověď:
1

Věta:
V sobotu ____ 36. ročník Karlovarského filmového festivalu.
Možnosti:
1) skončila
2) skončilo
3) skončili
4) skončily
5) skončil
Odpověď:
5

Věta:
Hodně zvláštní ____ i japonské sladkosti podávané na závěr.
Možnosti:
1) byla
2) bylo
3) byli
4) byly
5) byl
Odpověď:
4

Věta:
Pro vozy se ____ několik typů karosérií lišících se uspořádáním interiéru a vybavením.
Možnosti:
1) vyráběla
2) vyrábělo
3) vyráběli
4) vyráběly
5) vyráběl
Odpověď:
2

"""

request = """Doplň správně chybějící slovo do této věty:
Věta:
{sentence}
Možnosti:
1) {choices[0]}
2) {choices[1]}
3) {choices[2]}
4) {choices[3]}
5) {choices[4]}
Odpověď:
"""

input_variables = ["sentence", "choices"]
input_types = {"question": "str", "choices": "List[str]"}


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

