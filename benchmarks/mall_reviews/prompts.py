# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt_template = """Urči sentiment zadaného textu. Odpověz číslem 1 pro pozitivní sentiment, 0 pro neutrální sentiment, nebo -1 pro negativní sentiment.
Vždy odpovídej pouze tímto jedním číslem bez dalšího komentáře.

Zde je 5 ukázkových příkladů:

Text:
Miska neuvěřitelně páchne plastem, nepomohlo ani umytí v myčce.
Odpověď:
-1

Text:
Vyborna pochoutka pro psy!
Odpověď:
1

Text:
Mohu jen doporučit. Pokud netrváte na zvýšené odolnosti (prach, voda), tak je svým poměrem cena/kapacita/kompaktnost výborný! Také rychlost čtení/zápis není špatná. Samozřejmě např. flashdisky od výborné značky OCZ rychlostí příp. odolností nedoženou, ale ... záleží na prioritách.
Odpověď:
0

Text:
Jsem spokojena, přes kabel krásný obraz i zvuk , doporučuji :-)
Odpověď:
1

Text:
Trošku se mi nezdá přidělání gumičkami, nevím, jak bude trvanlivé, ale to se po ani ne měsíci nedá hodnotit. Škoda, že se nedají "vyhodit" z menu funkce dostupné po dokoupení (frekvence šlapání, druhé kolo). Jinak spokojenost, jen si musí člověk zvyknout, že to trvá tak 2-5 vteřin, než tachometr zareaguje na pohyb.
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

system_template = """Urči sentiment zadaného textu. Odpověz číslem 1 pro pozitivní sentiment, 0 pro neutrální sentiment, nebo -1 pro negativní sentiment.
Vždy odpovídej pouze tímto jedním číslem bez dalšího komentáře.

Zde je 5 ukázkových příkladů:

Text:
Miska neuvěřitelně páchne plastem, nepomohlo ani umytí v myčce.
Odpověď:
-1

Text:
Vyborna pochoutka pro psy!
Odpověď:
1

Text:
Mohu jen doporučit. Pokud netrváte na zvýšené odolnosti (prach, voda), tak je svým poměrem cena/kapacita/kompaktnost výborný! Také rychlost čtení/zápis není špatná. Samozřejmě např. flashdisky od výborné značky OCZ rychlostí příp. odolností nedoženou, ale ... záleží na prioritách.
Odpověď:
0

Text:
Jsem spokojena, přes kabel krásný obraz i zvuk , doporučuji :-)
Odpověď:
1

Text:
Trošku se mi nezdá přidělání gumičkami, nevím, jak bude trvanlivé, ale to se po ani ne měsíci nedá hodnotit. Škoda, že se nedají "vyhodit" z menu funkce dostupné po dokoupení (frekvence šlapání, druhé kolo). Jinak spokojenost, jen si musí člověk zvyknout, že to trvá tak 2-5 vteřin, než tachometr zareaguje na pohyb.
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

