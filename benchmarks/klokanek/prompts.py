# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Vyřeš zadanou úlohu výběrem jedné z 5 nabídnutých možností. Odpovídej vždy pouze písmenem odpovídajícím zvolené odpovědi.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Úloha:
Bedřich je o 1 rok a 1 den starší než Anežka. Narodil se 1. ledna 2002. Kdy se narodila Anežka?	
Možnosti:
A) 2. ledna 2003
B) 2. ledna 2001
C) 31. prosince 2000
D) 31. prosince 2002
E) 31. prosince 2003
Odpověď:
A

Úloha:
Anička a Bětka mají dohromady 10 bonbonů. Bětka jich má o 2 více než Anička. Kolik bonbonů má Bětka?
Možnosti:
A) 8
B) 6
C) 4
D) 2
E) 1
Odpověď:
B

Úloha:
V rovnici KAN - GAR = OO představují různá písmena různé číslice, stejná písmena stejné číslice. Najdětě největší možnou hodnotu čísla KAN.
Možnosti:
A) 987
B) 876
C) 865
D) 864
E) 785
Odpověď:
D

Úloha:
Kája, Eliška a Lucka slaví narozeniny ve stejný den. Jako každý rok dostaly společný dort, na kterém je napsán součet jejich věků. Letos je to 44. Které číslo tam bude napsáno příště, až to bude opět dvojmístné číslo zapsané týmiž číslicemi?
Možnosti:
A) 55
B) 66
C) 77
D) 88
E) 99
Odpověď:
C

Úloha:
Eva, Lucie a Magda spolu hrály turnaj v piškvorkách. Každé partie se účastnily právě dvě z těchto dívek, žádná neskončila remízou. Po každé partii nastoupila vítězka předchozí partie a dívka, která ji nehrála. Eva hrála celkem 10krát, Lucka 15krát a Magda 17krát. Kdo všechno mohl vyhrát druhou partii?
Možnosti:
A) Eva
B) Lucie
C) Magda
D) Eva nebo Magda
E) Lucie nebo Magda
Odpověď:
E

"""

request = """Vyřeš následující úlohu:
Úloha:
{question}
Možnosti:
A) {optionA}
B) {optionB}
C) {optionC}
D) {optionD}
E) {optionE}
Odpověď:
"""

input_variables = ["question", "optionA", "optionB", "optionC", "optionD", "optionE"]
input_types = {"question": "str", "optionA": "str", "optionB": "str", "optionC": "str", "optionD": "str", "optionE": "str"}


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

