# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


prompt_template = """Answer the given question by choosing one of the proposed answers.
Always answer only by the letter corresponding to the chosen answer without any further comment.

Here are 5 example questions:

Question:
Which factor will most likely cause a person to develop a fever?
Choices:
A) a leg muscle relaxing after exercise
B) a bacterial population in the bloodstream
C) several viral particles on the skin
D) carbohydrates being digested in the stomach
Answer:
B

Question:
When a switch is used in an electrical circuit, the switch can
Choices:
A) cause the charge to build.
B) increase and decrease the voltage.
C) cause the current to change direction.
D) stop and start the flow of current.
Answer:
D

Question:
Which of the following is an example of an assistive device?
Choices:
A) contact lens
B) motorcycle
C) raincoat
D) coffee pot
Answer:
A

Question:
A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
Choices:
A) has a pleasant flavor.
B) is inexpensive to produce.
C) neutralizes digestive acid.
D) occurs naturally in the body.
Answer:
C

Question:
Earth's core is primarily composed of which of the following materials?
Choices:
A) basalt
B) iron
C) magma
D) quartz
Answer:
B

Answer the following question:
Question:
{question}
Choices:
{choices}Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question", "choices"]
)

system_template = """Answer the given question by choosing one of the proposed answers.
Always answer only by the letter corresponding to the chosen answer without any further comment.

Here are 5 example questions:

Question:
Which factor will most likely cause a person to develop a fever?
Choices:
A) a leg muscle relaxing after exercise
B) a bacterial population in the bloodstream
C) several viral particles on the skin
D) carbohydrates being digested in the stomach
Answer:
B

Question:
When a switch is used in an electrical circuit, the switch can
Choices:
A) cause the charge to build.
B) increase and decrease the voltage.
C) cause the current to change direction.
D) stop and start the flow of current.
Answer:
D

Question:
Which of the following is an example of an assistive device?
Choices:
A) contact lens
B) motorcycle
C) raincoat
D) coffee pot
Answer:
A

Question:
A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate
Choices:
A) has a pleasant flavor.
B) is inexpensive to produce.
C) neutralizes digestive acid.
D) occurs naturally in the body.
Answer:
C

Question:
Earth's core is primarily composed of which of the following materials?
Choices:
A) basalt
B) iron
C) magma
D) quartz
Answer:
B
"""

msg_template = """Answer the following question:
Question:
{question}
Choices:
{choices}Answer:
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(msg_template),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

