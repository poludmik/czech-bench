# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Urči sentiment zadaného textu. Odpověz číslem 1 pro pozitivní sentiment, 0 pro neutrální sentiment, nebo -1 pro negativní sentiment.
Vždy odpovídej pouze tímto jedním číslem bez dalšího komentáře.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Text:
Rek bych ze mekac trosku nezachapal tvoji otazku :D
Odpověď:
0

Text:
Moc krásná fotečka :-)
Odpověď:
1

Text:
Já mám iPhone a nejde to!
Odpověď:
-1

Text:
parada, konecne si je zase jeden z velkych hracu vedom nastupujici budoucnosti. Diky
Odpověď:
1

Text:
jasně, že Vyskoká u Miskovic Kutná Hora :) hned kousíček je zřícenina Kláštera Belveder :)
Odpověď:
0

"""

request = """Vygeneruj klasifikaci pro následující příklad:
Text:
{text}
Odpověď:
"""

input_variables = ["text"]
input_types = {"text": "str"}


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

