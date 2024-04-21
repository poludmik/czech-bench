# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """For the given premise and hypothesis, decide whether the premise supports, refutes, or does not provide enough information for the hypothesis.
If the premise supports the hypothesis, return 0. If it refutes it, return 2. If it neither supports nor refutes it, return 1. Always respond with this single digit without any further commentary.

"""

few_shot = """Here are 5 examples:

Premise:
Earnings before interest and tax jumped to 4.55 billion euros ($5.34 billion) from 1.90 billion a year earlier, VW said in a statement on Thursday. "I am firmly convinced that our financial footing is adequate to cope with the transformation in the automotive industry and topics of the future," finance chief Frank Witter said in the statement.
Hypothesis:
VW wants to be part of the transformation in the automotive industry.
Label:
0

Premise:
A recent study found no evidence of seasonal affective disorder in Iceland where the sun does not appear for a long time in the winter.
Hypothesis:
The sun appears often in iceland during the winter
Label:
2

Premise:
Craig Conway, který byl propuštěn z funkce generálního ředitele společnosti PeopleSoft předtím, než společnost koupila společnost Oracle, byl minulý týden v Anglii.
Hypothesis:
Craig Conway was fired because he wanted to go to England last week
Label:
1

Premise:
Ordonez Reyes accused Jose Jesus Pena, alleged chief of security for the Nicaraguan embassy in Tegucigalpa, of masterminding the January 7th assassination of contra-commander Manuel Antonio Rugama.
Hypothesis:
Jose killed Ordonez and Manuel
Label:
2

Premise:
More than 150 dolphins, marine turtles and beaked whales have been washed up dead on beaches in Africa.
Hypothesis:
The government is warning people to stay away from the dead bodies
Label:
1

"""

request = """Generate the label for the following example:
Premise:
{context}
Hypothesis:
{claim}
Label:
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

