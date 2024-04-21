# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Pro zadaný kontext a s ním souvisejícím otázku vygeneruj správnou odpověď. Odpovídej minimálním počtem slov extrahovaných z kontextu bez dalšího komentáře.
Pokud kontext neobsahuje odpověď na danou otázku, odpověz pouze znakem '-' a dále se nevyjadřuj.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Kontext:
V osmi letech se Beyoncé a kamarádka z dětství Kelly Rowlandová seznámila s LaTavií Robersonovou na konkurzu do dívčí zábavní skupiny. Byli zařazeni do skupiny se třemi dalšími dívkami jako Girl's Tyme a rapovali a tančili na talentové show v Houstonu. Po zhlédnutí skupiny je producent R&B Arne Frager přivezl do svého studia v severní Kalifornii a umístil je do Star Search, největší talentové show v té době v celostátní televizi. Girl's Tyme nevyhrála a Beyoncé později řekla, že píseň, kterou předvedli, nebyla dobrá. V roce 1995 Beyoncéin otec rezignoval na svou práci, aby skupinu řídil. Stěhování snížilo příjem rodiny Beyoncé o polovinu a její rodiče byli nuceni přestěhovat se do oddělených bytů. Mathew zkrátil původní sestavu na čtyři a skupina pokračovala v vystupování jako předskokan pro další zavedené dívčí skupiny R&B. Dívky se přihlásily do konkurzu před nahrávací společnosti a nakonec se upsaly Elektra Records, krátce se přestěhovaly do Atlanta Records, aby pracovaly na své první nahrávce, jenže pak je společnost vystřihla. To rodinu ještě více zatížilo a rodiče Beyoncé se rozešli. 5. října 1995 podepsal Dwayne Wiggins's Grass Roots Entertainment smlouvu se skupinou. V roce 1996 začaly dívky na základě dohody se Sony Music nahrávat své debutové album, rodina Knowlesových se znovu spojila a krátce nato skupina uzavřela smlouvu s Columbia Records.
Otázka:
Kdo byl první nahrávací společnost, která dala holkám nahrávací smlouvu?
Odpověď:
Elektra Records

Kontext:
Čaj Longjing (také nazývaný čaj z dračí studny), pocházející z Chang-čou, je jedním z nejprestižnějších, ne-li nejprestižnějších čínských čajů. Hangzhou je také proslulé svými hedvábnými deštníky a ručními vějíři. Kuchyně Zhejiang (sama rozdělená do mnoha tradic, včetně kuchyně Chang-čou) je jednou z osmi velkých tradic čínské kuchyně.
Otázka:
Kde je zakázaný čaj Longjing?
Odpoveď:
-

Kontext:
Dále jsou zde tři zástupci náčelníků štábů obrany se zvláštními pravomocemi, zástupce náčelníka štábu obrany (schopnost), zástupce CDS (personál a výcvik) a zástupce CDS (operace). Hlavní lékař, zastupuje Zdravotnickou službu obrany v rámci personálu obrany a je klinickým vedoucím této služby.
Otázka:
Kolik je tam zástupců náčelníků štábů obrany?
Odpověď:
tři

Kontext:
Německá říše dobyla Ukrajinu během první světové války a plánovala ji buď anektovat, nebo dosadit loutkového krále, ale byla poražena Ententem, s velkou účastí ukrajinských bolševiků. Poté, co Ukrajina dobyla zbytek Ukrajiny od Bělochů, připojila se k SSSR a byla rozšířena (získala Krym a poté Východní Galicii), načež byl za podpory Moskvy zahájen proces Ukrajinizace.
Otázka:
Během které války Ukrajina dobyla Německou říši?
Odpověď:
-

Kontext:
Německá říše dobyla Ukrajinu během první světové války a plánovala ji buď anektovat, nebo dosadit loutkového krále, ale byla poražena Ententem, s velkou účastí ukrajinských bolševiků. Poté, co Ukrajina dobyla zbytek Ukrajiny od Bělochů, připojila se k SSSR a byla rozšířena (získala Krym a poté Východní Galicii), načež byl za podpory Moskvy zahájen proces Ukrajinizace.
Otázka:
Která ukrajinská politická skupina se podílela na porážce německého impéria?
Odpověď:
bolševiků

"""

request = """Nyní vygeneruj odpověď pro následující zadání:
Kontext:
{context}
Otázka::
{question}
Odpověď:
"""

input_variables = ["context", "question"]
input_types = {"context": "str", "question": "str"}


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

