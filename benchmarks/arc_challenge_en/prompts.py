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
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
Choices:
A) dry palms
B) wet palms
C) palms covered with oil
D) palms covered with lotion
Answer:
A

Question:
Which of the following statements best explains why magnets usually stick to a refrigerator door?
Choices:
A) The refrigerator door is smooth.
B) The refrigerator door contains iron.
C) The refrigerator door is a good conductor.
D) The refrigerator door has electric wires in it.
Answer:
B

Question:
A fold observed in layers of sedimentary rock most likely resulted from the
Choices:
A) cooling of flowing magma.
B) converging of crustal plates.
C) deposition of river sediments.
D) solution of carbonate minerals.
Answer:
B

Question:
As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?
Choices:
A) 31 pounds and 14 kilograms
B) 31 pounds and 84 kilograms
C) 186 pounds and 14 kilograms
D) 186 pounds and 84 kilograms
Answer:
D

Question:
Which of the following areas is most likely to form metamorphic rocks such as gneiss and schist?
Choices:
A) a sea floor
B) a windblown desert
C) a site deep underground
D) a site covered by a glacier
Answer:
C

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
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
Choices:
A) dry palms
B) wet palms
C) palms covered with oil
D) palms covered with lotion
Answer:
A

Question:
Which of the following statements best explains why magnets usually stick to a refrigerator door?
Choices:
A) The refrigerator door is smooth.
B) The refrigerator door contains iron.
C) The refrigerator door is a good conductor.
D) The refrigerator door has electric wires in it.
Answer:
B

Question:
A fold observed in layers of sedimentary rock most likely resulted from the
Choices:
A) cooling of flowing magma.
B) converging of crustal plates.
C) deposition of river sediments.
D) solution of carbonate minerals.
Answer:
B

Question:
As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?
Choices:
A) 31 pounds and 14 kilograms
B) 31 pounds and 84 kilograms
C) 186 pounds and 14 kilograms
D) 186 pounds and 84 kilograms
Answer:
D

Question:
Which of the following areas is most likely to form metamorphic rocks such as gneiss and schist?
Choices:
A) a sea floor
B) a windblown desert
C) a site deep underground
D) a site covered by a glacier
Answer:
C
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

