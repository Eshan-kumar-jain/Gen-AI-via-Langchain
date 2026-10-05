import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())

template = PromptTemplate(
    template = "Generate 5 interesting points about {topic}",
    input_variables = ['input']
)


parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic': 'India'})

print(result)

chain.get_graph().print_ascii()