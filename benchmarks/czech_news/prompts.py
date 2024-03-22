# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Pro zadaný text pocházející ze zpravodajského článku urči jeho kategorii z následujícího výběru:
1) Zahraniční
2) Domácí
3) Sport
4) Kultura
5) Ekonomika
Vždy vracej pouze číslo kategorie bez dalšího komentáře.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Text:
Ohňostroji, veselicemi a s jásotem přivítal svět příchod roku 2000. Jako poslední obyvatelé Země sledovali západ Slunce v roce 1999 obyvatelé tichomořského zámořského území USA, Americké Samoy. Slunce se naposledy v roce 1999 schovalo za duhovým horizontem v pátek 06:57 večer místního času (tj. v sobotu v 07:57 SEČ). Petardy a bengálské ohně usmrtily v Evropě šest lidí a desítky dalších zranily. Oslavy i přesto nepotvrdily chmurné představy policistů, kteří čekali daleko větší potíže a byli ve zvýšené pohotovosti ve všech evropských metropolích
Klasifikace:
1

Text:
V nesmírně vyrovnané tabulce fotbalové ligy má pátá Ostrava náskok jen sedmi bodů před posledním Žižkovem. Většinu týmů za Ostravou čeká tuhý boj o záchranu, a tak už se na něj připravují	
Klasifikace:
3

Text:
Strana, která by vznikla ze studentské výzvy "Děkujeme, odejděte!", by nyní mohla vyhrát případné parlamentní volby. Na její kandidaturu by přitom nejvíce doplatily Unie svobody a ODS. Účast nové strany by navíc zvýšila předpokládanou volební účast o šest procent na 85,7 procenta občanů. Vyplývá to z exkluzivního prosincového šetření agentury Sofres-Factum pro ČTK. "Studentskou" stranu by těsně před Vánocemi volilo 24,7 procenta občanů. Na druhém místě by skončila ODS, které by dalo hlas 14,4 procenta dotázaných
Klasifikace:
2

Text:
Tržby maloobchodních organizací vzrostly za celý loňský rok o 2,1 procenta, v samotném prosinci to bylo dokonce o 6,4 procenta. Letošní výhledy jsou podle ekonomů mírně horší, tržby maloobchodu nají podle jejich odhadů vzrůst přibližně o procento. Vliv bude mít nižší růst reálných mezd, které loni dosáhly vysokých 6 procent, na tržbách se podepíše se i vyšší nezaměstnanost	
Klasifikace:
5

Text:
Třináctého ledna odstartuje v pražské Městské knihovně další, již šestá část unikátní filmové přehlídky Projekt 100. Společná akce Asociací českých a slovenských filmových klubů a Městských divadel v Uherském Hradišti se zrodila před čtyřmi lety při příležitosti stého výročí vzniku kinematografie, nyní se s šestou desítkou filmů dostává již do své druhé poloviny
Klasifikace:
4

"""

request = """Vygeneruj klasifikaci pro následující příklad:
Text:
{brief}
Klasifikace:
"""

input_variables = ["brief"]
input_types = {"brief": "str"}


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

