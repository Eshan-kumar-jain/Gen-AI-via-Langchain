import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative","other"] = Field(description="The sentiment classification of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write a precise response for the positive feedback and also give which feedback it is \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write a precise response for the negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == "positive", prompt2 | model | parser1),
    (lambda x: x['sentiment'].sentiment == "negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "Invalid sentiment classification")
)

# Keeps 'feedback' and adds 'sentiment' -> {'feedback': '...', 'sentiment': Feedback(...)}
chain = RunnablePassthrough.assign(sentiment=classifier_chain) | branch_chain

feedback = input("Enter your feedback: ")

result = chain.invoke({'feedback': feedback})

print(result)

chain.get_graph().print_ascii()