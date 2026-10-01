import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age, city of a fictional person \n {format_instructions}",
    input_variables = [],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

#prompt = template.format()

#result = model.invoke(prompt)

#final_result = parser.parse(result.content)

#print(final_result)

#print(type(final_result))

chain = template | model | parser

result = chain.invoke({})

print(result)
print(type(result))