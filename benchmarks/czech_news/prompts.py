# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt_template = """Pro zadaný text pocházející ze zpravodajského článku urči jeho kategorii z následujícího výběru:

1: Zahraniční
2: Domácí
3: Sport
4: Kultura
5: Revue
6: Koktejl
7: Ekonomika
8: Krimi
9: Podnikání
10: Auto
11: Věda
12: Komentáře
13: Cestování
14: Finance
15: Technologie
16: Bydlení
17: Koronavirus
18: Byznys
19: Rozhovory
20: Podcasty
21: Životní styl
22: Literatura
23: Vánoce
24: Výtvarné umění
25: Kolo

Vždy vracej pouze číslo kategorie bez dalších komentářů.

Zde je 5 ukázkových příkladů:

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
Slovenský prezident Rudolf Schuster po jednání se svým čínským partnerem Ťiang-Ce-minem daroval čínskému prezidentovi valašku, kterou ke zděšení ochranky propašoval do přísně střeženého prezidentského paláce	
Klasifikace:
6

Text:
Moderátorka a spolumajitelka Divadla bez zábradlí Hana Heřmánková spojila příjemné s užitečným a pozvala na jeviště známé herecké a moderátorské tváře, aby představily originální šaty Beaty Rajské, Táni Vokřálové a značky La Halle a Tveret. "Konečně život, to je přesně to, co na molech chybí," komentovala atmosféru módní návrhářka Beata Rajská. Celebrity totiž, nejen že šaty obléky a ukázaly, ale zároveň - díky choreografii - působily velmi přirozeně. Herce a moderátory v rolích modelek a modelů najdete v bohaté fotogalerii
Klasifikace:
5

Vygeneruj klasifikaci pro následující příklad:
Text:
{brief}
Klasifikace:

"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["brief"]
)

system_template = """Pro zadaný text pocházející ze zpravodajského článku urči jeho kategorii z následujícího výběru:

1: Zahraniční
2: Domácí
3: Sport
4: Kultura
5: Revue
6: Koktejl
7: Ekonomika
8: Krimi
9: Podnikání
10: Auto
11: Věda
12: Komentáře
13: Cestování
14: Finance
15: Technologie
16: Bydlení
17: Koronavirus
18: Byznys
19: Rozhovory
20: Podcasty
21: Životní styl
22: Literatura
23: Vánoce
24: Výtvarné umění
25: Kolo

Vždy vracej pouze číslo kategorie bez dalších komentářů.

Zde je 5 ukázkových příkladů:

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
Slovenský prezident Rudolf Schuster po jednání se svým čínským partnerem Ťiang-Ce-minem daroval čínskému prezidentovi valašku, kterou ke zděšení ochranky propašoval do přísně střeženého prezidentského paláce	
Klasifikace:
6

Text:
Moderátorka a spolumajitelka Divadla bez zábradlí Hana Heřmánková spojila příjemné s užitečným a pozvala na jeviště známé herecké a moderátorské tváře, aby představily originální šaty Beaty Rajské, Táni Vokřálové a značky La Halle a Tveret. "Konečně život, to je přesně to, co na molech chybí," komentovala atmosféru módní návrhářka Beata Rajská. Celebrity totiž, nejen že šaty obléky a ukázaly, ale zároveň - díky choreografii - působily velmi přirozeně. Herce a moderátory v rolích modelek a modelů najdete v bohaté fotogalerii
Klasifikace:
5
"""

msg_template = """Vygeneruj klasifikaci pro následující příklad:
Text:
{brief}
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

