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
Praha 14. července (ČTK) - Sedmnáct let byla členkou pražského Národního divadla a její herectví hojně využíval také film. Regina Rázlová hrála většinou sebevědomé a exkluzivní ženy, často i na opačné straně zákona, a její podmanivý hlas zněl i v rozhlase a dabingu. Po roce 1989 se jí přestalo dařit, dala se na podnikání a několik měsíců dokonce strávila ve vazbě kvůli kauze vytunelovaného Skloexportu. K tomu se přidaly zdravotní problémy. Rázlová ale v duchu hrdinek, která ztvárňovala v divadle i ve filmu, vše špatné dokázala překonat. Dnes je Rázlová, která 16. července oslaví sedmdesátiny, opět žádanou herečkou a celebritou.
Tvrzení:
Regina Rázlová je zpěvačka.
Klasifikace:
0

Kontext:
Washington 20. srpna (ČTK) - Přední kongresman Demokratické strany Barney Frank se tento týden nečekaně vyslovil pro zrušení zestátněných hypotečních agentur Fannie Mae a Freddie Mac, které významně přispěly ke vzniku a prohloubení těžké finanční krize v USA z let 2008-09. Od vlivného spoluautora rozsáhlého zákona o reformě finanční regulace to je obrat o 180 stupňů, protože Frank dříve nad oběma agenturami držel podle pozorovatelů ochrannou ruku.
Tvrzení:
Kongresman Frank chce zrušit dvě agentury.
Klasifikace:
2

Kontext:
Prodej živých delfínů se řídí dohodou o mezinárodním obchodu s ohroženými druhy , která zakazuje podobné transakce , pokud by mohly zvířatům uškodit . Šalamounovy ostrovy , ležící asi 1800 kilometrů severovýchodně od Austrálie , nicméně dohodu nepodepsaly . Území je v současné době zmítáno politickou krizí a etnickými násilnostmi , kvůli nimž sem byli tento týden vysláni australští vojáci . Ekologové viní mexické podnikatele , že krize na Šalamounových ostrovech zneužili .
Tvrzení:
Austrálie odmítla dohodu o mezinárodním obchodu s ohroženými druhy týkající se například prodeje živých delfínů.
Klasifikace:
1

Kontext:
BRNO/BUČOVICE (Vyškovsko) 1. června (ČTK) - Padesát let restaurování gobelínů v Moravské gobelínové manufaktuře ve Valašském Meziříčí na Vsetínsku přibližuje ode dneška výstava v zámku Bučovice na Vyškovsku. Potrvá do konce srpna, řekl dnes ČTK bučovický kastelán Pavel Ecler. Manufaktura do bučovického zámku přivezla mimo jiné i šest historických gobelínů z depozitářů státního zámku Náměšť nad Oslavou a Slezského zemského muzea v Opavě.
Tvrzení:
Na zámku Bučovice není výstava o restaurování gobelínů.
Klasifikace:
0

Kontext:
Na 3200 kilometrů dlouhé jižní hranici USA bylo loni zadrženo 1,1 milionu nelegálních přistěhovalců z Mexika , z toho více než polovina na 600 kilometrech v arizonské pohraniční poušti . Ministerstvo vnitřní bezpečnosti chystá na jaře , kdy imigrační vlna vrcholí , posílit ostrahu o dalších 500 lidí na 2500 pohraničníků a dočasně tam převelet 27 letounů .
Tvrzení:
Migrační vlna z Mexika do USA vrcholí v dubnu.
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

