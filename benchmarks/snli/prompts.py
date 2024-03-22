# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Pro zadaný kontext a testované tvrzení rozhodni, zda kontext potrzuje obsah tvrzení, popírá ho, nebo neobsahuje dostatečné informace. 
Pokud tvrzení potvrzuje, vrať číslo 2. Pokud jej vyvrací, vrať číslo 0. Pokud nelze rozhodnout, vrať číslo 1. Vždy odpovídej pouze touto jednou číslicí bez dalšího komentáře.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Kontext:
Lidé na kolech čekají na křižovatce.
Tvrzení:
Právě teď se konají cyklistické závody.
Klasifikace:
1

Kontext:
Děti jdou domů ze školy.
Tvrzení:
Děti jsou v knihovně.
Klasifikace:
0

Kontext:
Děti se usmívají a mávají na kameru.
Tvrzení:
Jsou tam děti.
Klasifikace:
2

Kontext:
Starší muž pije pomerančový džus v restauraci.
Tvrzení:
Muž pije džus.
Klasifikace:
2

Kontext:
Bílý kůň táhne vozík, zatímco muž stojí a sleduje.
Tvrzení:
Kůň tahá zboží.
Klasifikace:
1

"""

request = """Vygeneruj klasifikaci pro následující příklad:
Kontext:
{context}
Tvrzení:
{claim}
Klasifikace:
"""

input_variables = ["context", "claim"]
input_types = {"context": "str", "claim": "str"}


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

