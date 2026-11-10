from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template = 'Summarize the following pdf \n {pdf}',
    input_variables = ['pdf']
)

parser = StrOutputParser()

loader = PyPDFLoader('Eshan_Kumar_Jain_AI_ML_Engineer_Resume (1).pdf')

docs = loader.load()

print(type(docs))

#print(docs[0])

print(type(docs[0]))

chain = prompt | model | parser

result = chain.invoke({'pdf': docs[0].page_content})

print(result)