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
People on bicycles waiting at an intersection.
Hypothesis:
There is a bike race happening right now.
Label:
1

Premise:
Children going home from school.
Hypothesis:
The children are at the library.
Label:
2

Premise:
Children smiling and waving at camera.
Hypothesis:
There are children present.
Label:
0

Premise:
An older man is drinking orange juice at a restaurant.
Hypothesis:
A man is drinking juice.
Label:
0

Premise:
A white horse is pulling a cart while a man stands and watches.
Hypothesis:
A horse is hauling goods.
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

