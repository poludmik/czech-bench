# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt_template = """Pro zadaný kontext a testované tvrzení rozhodni, zda kontext potrzuje obsah tvrzení, popírá ho, nebo neobsahuje dostatečné informace. 
Pokud tvrzení potvrzuje, vrať číslo 2. Pokud jej vyvrací, vrať číslo 0. Pokud nelze rozhodnout, vrať číslo 1. Vždy odpovídej pouze touto jednou číslicí bez dalšího komentáře.

Zde je 5 ukázkových příkladů:

Kontext:
Zisk před úroky a zdaněním vyskočil na 4,55 miliardy eur (5,34 miliardy dolarů) z 1,90 miliardy o rok dříve, uvedl VW ve čtvrtečním prohlášení. „Jsem pevně přesvědčen, že naše finanční základna je dostatečná k tomu, abychom zvládli transformaci v automobilovém průmyslu a témata budoucnosti,“ uvedl ve svém prohlášení šéf financí Frank Witter. 
Tvrzení:
VW chce být součástí transformace v automobilovém průmyslu.
Klasifikace:
2

Kontext:
Nedávná studie nenalezla žádné důkazy sezónní afektivní poruchy na Islandu, kde se slunce v zimě dlouho neobjevuje.
Tvrzení:
Slunce se na Islandu v zimě dlouho objevuje.
Klasifikace:
0

Kontext:
Craig Conway, který byl propuštěn z funkce generálního ředitele společnosti PeopleSoft předtím, než společnost koupila společnost Oracle, byl minulý týden v Anglii.
Tvrzení:
Craig Conway byl propuštěn, protože chtěl minulý týden jet do Anglie
Klasifikace:
1

Kontext:
Ordonez Reyes obvinil Joseho Jesusa Penu, údajného šéfa bezpečnosti nikaragujského velvyslanectví v Tegucigalpě, ze zorganizování atentátu na velitele Manuela Antonia Rugamu 7. ledna.	
Tvrzení:
Jose zabil Ordoneze a Manuela
Klasifikace:
0

Kontext:
Více než 150 delfínů, mořských želv a zobáků velryb bylo vyplaveno na mrtvé pláže v Africe.	
Tvrzení:
Vláda varuje lidi, aby se drželi dál od mrtvých těl
Klasifikace:
1

Vygeneruj klasifikaci pro následující příklad:
Kontext:
{context}
Tvrzení:
{claim}
Klasifikace:

"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "claim"]
)

system_template = """Pro zadaný kontext a testované tvrzení rozhodni, zda kontext potrzuje obsah tvrzení, popírá ho, nebo neobsahuje dostatečné informace. 
Pokud tvrzení potvrzuje, vrať číslo 2. Pokud jej vyvrací, vrať číslo 0. Pokud nelze rozhodnout, vrať číslo 1. Vždy odpovídej pouze touto jednou číslicí bez dalšího komentáře.

Zde je 5 ukázkových příkladů:

Kontext:
Zisk před úroky a zdaněním vyskočil na 4,55 miliardy eur (5,34 miliardy dolarů) z 1,90 miliardy o rok dříve, uvedl VW ve čtvrtečním prohlášení. „Jsem pevně přesvědčen, že naše finanční základna je dostatečná k tomu, abychom zvládli transformaci v automobilovém průmyslu a témata budoucnosti,“ uvedl ve svém prohlášení šéf financí Frank Witter. 
Tvrzení:
VW chce být součástí transformace v automobilovém průmyslu.
Klasifikace:
2

Kontext:
Nedávná studie nenalezla žádné důkazy sezónní afektivní poruchy na Islandu, kde se slunce v zimě dlouho neobjevuje.
Tvrzení:
Slunce se na Islandu v zimě dlouho objevuje.
Klasifikace:
0

Kontext:
Craig Conway, který byl propuštěn z funkce generálního ředitele společnosti PeopleSoft předtím, než společnost koupila společnost Oracle, byl minulý týden v Anglii.
Tvrzení:
Craig Conway byl propuštěn, protože chtěl minulý týden jet do Anglie
Klasifikace:
1

Kontext:
Ordonez Reyes obvinil Joseho Jesusa Penu, údajného šéfa bezpečnosti nikaragujského velvyslanectví v Tegucigalpě, ze zorganizování atentátu na velitele Manuela Antonia Rugamu 7. ledna.	
Tvrzení:
Jose zabil Ordoneze a Manuela
Klasifikace:
0

Kontext:
Více než 150 delfínů, mořských želv a zobáků velryb bylo vyplaveno na mrtvé pláže v Africe.	
Tvrzení:
Vláda varuje lidi, aby se drželi dál od mrtvých těl
Klasifikace:
1
"""

msg_template = """Vygeneruj klasifikaci pro následující příklad:
Kontext:
{context}
Tvrzení:
{claim}
Klasifikace:

"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(msg_template),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

