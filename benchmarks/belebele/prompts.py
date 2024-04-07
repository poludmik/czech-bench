# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Zodpověz zadanou otázku výběrem jedné z nabízených možností. Odpovídej na základě poskytnutého kontextu.
Zvolenou odpověď neopakuj. Vždy odpovídej pouze číslem odpovídajícím vybrané odpovědi bez dalšího komentáře.

"""

few_shot = """Zde je 5 ukázkových příkladů:

Kontext:
Ujistěte se, že máte ruku maximálně uvolněnou, přitom správně mačkejte všechny klávesy – také se snažte prsty příliš nehýbat kolem. Tímto způsobem se unavíte co nejméně. Nezapomeňte, že pro větší hlasitost není třeba tisknout klávesy velkou silou jako v případě klavíru. Pokud chcete na akordeon hrát hlasitěji, musíte měch stlačovat silněji nebo rychleji.
Otázka:
Co podle této pasáže není považováno za přesný tip pro správnou hru na akordeon?
Možnosti:
1) Chcete-li zvýšit hlasitost, zvyšte sílu úderu prstů do kláves
2) Za účelem zachování energie je třeba co nejvíce omezovat zbytečné pohyby
3) Uvědomujte si, jaké klávesy mačkáte, a přitom mějte ruku uvolněnou
4) Za účelem zvýšení hlasitosti hry zvyšte rychlost mačkání měchů
Odpověď:
1

Kontext:
Jedním z nejčastějších problémů při pokusu konvertovat film do formátu DVD je takzvaný overscan. Většina televizí je vyrobena tak, aby se líbila široké veřejnosti. Z tohoto důvodu má vše, co vidíte v televizi, oříznuté kraje nahoře, dole i po stranách. Je to pro ujištění, že obrázek zakrývá celou obrazovku. To se nazývá overscan. Bohužel, když vytváříte video DVD, okraje videa budou ořezané také, a pokud jsou titulky videa příliš blízko spodnímu okraji, neukáží se celé.
Otázka:
Se kterými z následujících problémů se podle této pasáže člověk může setkat, když převádí film na DVD formát?
Možnosti:
1) Obraz není roztažený na celou obrazovku
2) Částečně ořezané titulky
3) Obraz je roztažený na celou obrazovku
4) Ořezané okraje
Odpověď:
2

Kontext:
Po přijetí Kongresem 4. července pak byl ručně psaný návrh, který podepsal Předseda kongresu John Hancock a tajemník Charles Thomson, poslán o pár bloků dál do tiskařství Johna Dunlapa. Přes noc bylo vyrobeno něco mezi 150 a 200 kopiemi, které jsou v současnosti nazývány „Dunlapovy výtisky“. Poprvé byl dokument veřejně předčítán Johnem Nixonem 8. července na nádvoří Síně nezávislosti. Jeden byl 6. července zaslán Georgi Washingtonovi, který jej přečetl svým vojákům v New Yorku 9. července. Do Londýna se jedna z kopií dostala 10. srpna. Nejstarší známé dochované exempláře dokumentu jsou tzv. Dunlap broadsides, neboli celkem 25 výtisků z Dunlapovy tiskárny. Originální rukopisný exemplář se nedochoval.
Otázka:
Čí podpis se objevil na ručně sepsaném konceptu?
Možnosti:
1) Johna Dunlapa
2) George Washingtona
3) Johna Nixona
4) Charlese Thomsona
Odpověď:
4

Kontext:
Plán Američanů spočíval v tom, že koordinovaně zaútočí ze tří různých směrů. Generál John Cadwalder měl spustit útok proti britské posádce v Bordentownu za účelem odpoutání pozornosti, aby byly odříznuty od jakýchkoliv posil. Generál James Ewing dovede 700 milicí přes řeku na Trenton Ferry, obsadí most přes Assunpink Creek a zabrání nepřátelským jednotkám v útěku. Hlavní útočná síla 2 400 mužů překročila řeku devět mil severně od Trentonu a poté se rozdělila na dvě skupiny (jednu pod Greenem a druhou pod Sullivanem), aby zahájila útok před úsvitem.
Otázka:
Kde se tam nacházela britská posádka?
Možnosti:
1) V Assunpink Creek
2) V Trentonu
3) V Bordentownu
4) V Princetonu
Odpověď:
3

Kontext:
Při pohledu na tuto činnost zavolali posily i Kolonisté. Vojáci posilující čelní pozice zahrnovali 1. a 3. pluk New Hampshire čítající 200 mužů pod velením plukovníků Johna Starka a Jamese Reeda (oba se později stali generály). Starkovi lidé obsadili pozice podél plotu na severním konci pozice kolonistů. Když odliv zpřístupnil prostor podél řeky Mystic na severovýchodě pevniny, rychle prodloužili plot o krátkou kamennou zeď k severnímu okraji na pokraj vodní plochy na malé pláži. Gridley nebo Stark umístili kůl zhruba 100 stop (30 m) před plot a přikázali, že nikdo nesmí vypálit, dokud jej protivníci neminou.
Otázka:
Kdy podle dané pasáže prodloužili Starkovi muži plot?
Možnosti:
1) Když kolonisté zavolali posily
2) Poté, co protivníci minuli kůl
3) Během odlivu
4) Během toho, co vojáci zaujímali pozice vepředu
Odpověď:
3

"""

request = """Odpověz na následující otázku:
Kontext:
{context}
Otázka:
{question}
Možnosti:
1) {option1}
2) {option2}
3) {option3}
4) {option4}
Odpověď:
"""

input_variables = ["context", "question", "option1", "option2", "option3", "option4"]


prompt_template = task + few_shot + request
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=input_variables
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

