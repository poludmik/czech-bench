# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt_template = """Urči zda zadaný výrok vyjadřuje subjektivní názor, nebo objektivní skutečnost. Odpověz číslem 0 pro subjektivní názor, nebo 1 pro objektivní skutečnost.
Vždy odpovídej pouze tímto jedním číslem bez dalšího komentáře.

Zde je 5 ukázkových příkladů:

Text:
Stejně vynikající jako jednička s ještě spletitějším dějem a s dokonalou vizí budoucnosti.	
Odpověď:
0

Text:
A tak se odvíjí příběhy 4 krásných slečen, které mají zdánlivě vše i svá tajemství, jenž by neměla být objevena.	
Odpověď:
1

Text:
Na muj vkus az moc "kecaci" a malo akce.	
Odpověď:
0

Text:
Dokument zachycuje příběhy čtyř lidí z Noveldy, kteří bojují za splnění svých snů přesto, že se setkávají se spoustou obtíží.	
Odpověď:
1

Text:
A musím se přiznat, že sem ani jednou neusnul, ačkoli je to bezmála dvouhodinová nuda jako prase.	
Odpověď:
0

Vygeneruj klasifikaci pro následující příklad:
Text:
{text}
Odpověď:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["text"]
)

system_template = """Urči zda zadaný výrok vyjadřuje subjektivní názor, nebo objektivní skutečnost. Odpověz číslem 0 pro subjektivní názor, nebo 1 pro objektivní skutečnost.
Vždy odpovídej pouze tímto jedním číslem bez dalšího komentáře.

Zde je 5 ukázkových příkladů:

Text:
Stejně vynikající jako jednička s ještě spletitějším dějem a s dokonalou vizí budoucnosti.	
Odpověď:
0

Text:
A tak se odvíjí příběhy 4 krásných slečen, které mají zdánlivě vše i svá tajemství, jenž by neměla být objevena.	
Odpověď:
1

Text:
Na muj vkus az moc "kecaci" a malo akce.	
Odpověď:
0

Text:
Dokument zachycuje příběhy čtyř lidí z Noveldy, kteří bojují za splnění svých snů přesto, že se setkávají se spoustou obtíží.	
Odpověď:
1

Text:
A musím se přiznat, že sem ani jednou neusnul, ačkoli je to bezmála dvouhodinová nuda jako prase.	
Odpověď:
0
"""

msg_template = """Vygeneruj klasifikaci pro následující příklad:
Text:
{text}
Odpověď:
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(msg_template),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

