# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """For the given statement, determine whether the statement conveys a subjective opinion or objective facts. Respond with the number 0 for subjective opinion, or 1 for objective facts.
Always respond only with this single digit without any further commentary.

"""

few_shot = """Here are 5 examples:

Text:
As excellent as the first, with an even more intricate plot and a perfect vision of the future.
Answer:
0

Text:
And so unfolds the story of four beautiful young women who seemingly have everything, including secrets that should not be discovered.
Answer:
1

Text:
Too much talk and too little action to my taste.	
Answer:
0

Text:
It documents the stories of four people from Novelda who are fighting to make their dreams come true, despite facing many obstacles.	
Answer:
1

Text:
And I have to admit that I did not fall asleep even once, although it is almost a two-hour borefest.	
Answer:
0

"""

request = """Classify the following example:
Text:
{text}
Answer:
"""

input_variables = ["text"]
input_types = {"text": "str"}


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

