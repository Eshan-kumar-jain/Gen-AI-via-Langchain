from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template = ' write a short summary for the following text \n {text}',
    input_variables = ['poem']
)

parser = StrOutputParser()

loader = TextLoader('sample.txt',encoding = "utf-8")

docs = loader.load()

print(type(docs))

#print(docs[0])

print(type(docs[0]))

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)