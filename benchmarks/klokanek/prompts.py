# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt_template = """Vyřeš zadanou úlohu výběrem jedné z 5 nabídnutých možností. Odpovídej vždy pouze písmenem odpovídajícím zvolené odpovědi.

Zde je 5 ukázkových příkladů:

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

Vygeneruj klasifikaci pro následující příklad:
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
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question", "optionA", "optionB", "optionC", "optionD", "optionE"]
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

