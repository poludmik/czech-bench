# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Odpověz na zadanou otázku výběrem jedné z nabídnutých možností.
Odpovídej vždy pouze písmenem odpovídajícím zvolené odpovědi bez dašího komentáře.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Otázka:
Vysokotlaké systémy zabraňují vzduchu stoupat do chladnějších oblastí atmosféry, kde může kondenzovat voda. Jaký bude nejpravděpodobnější výsledek, pokud se systém vysokého tlaku udrží v oblasti po dlouhou dobu?
Možnosti:
A) mlha
B) Déšť
C) sucho
D) Tornádo
Odpověď:
C

Otázka:
Která oblast by byla nejlepší pro výzkum, aby se našly způsoby, jak snížit problémy životního prostředí způsobené lidmi?
Možnosti:
A) Přeměna slunečního světla na elektřinu
B) Hledání nových zásob uhlí
C) Nalezení ložisek, která obsahují ropu
D) Přeměna lesů na zemědělskou půdu
Odpověď:
A

Otázka:
Jak jsou částice v bloku železa ovlivněny, když je blok roztaven?
Možnosti:
A) Částice získávají hmotnost.
B) Částice obsahují méně energie.
C) Částice se pohybují rychleji.
D) Částice se zvětšují v objemu.
Odpověď:
C

Otázka:
Každý rok je vykáceno přibližně 50 milionů akrů tropického deštného pralesa. Jaký efekt by nejpravděpodobněji vyplynul z vykácení těchto lesů?
Možnosti:
A) snížení eroze půdy
B) Pokles biodiverzity
C) zlepšení kvality ovzduší
D) Zlepšení kvality vody
Odpověď:
B

Otázka:
I když patří do stejné čeledi, orel a pelikán se liší. Jaký je mezi nimi rozdíl?
Možnosti:
A) Jejich preference pro konzumaci ryb
B) Schopnost létat
C) Způsob jejich rozmnožování
D) Způsob jejich chytání potravy
Odpověď:
D

"""

request = """Odpověz na následující otázku:
Otázka:
{question}
Možnosti:
{choices}Odpověď:
"""

input_variables = ["question", "choices"]
input_types = {"question": "str", "choices": "str"}


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

