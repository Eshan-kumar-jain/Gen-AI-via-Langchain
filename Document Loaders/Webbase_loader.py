from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template = 'Answer the following questions\n {question} from the foloowing text \n {text}',
    input_variables = ['question','text']
)

parser = StrOutputParser()

url = "https://en.wikipedia.org/wiki/Lionel_Messi"

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))

#print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'question':'What is the speciality of this person?','text':docs[0].page_content})
print(result)