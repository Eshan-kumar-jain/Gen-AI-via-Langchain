from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template = 'Answer the following questions\n {question} from the foloowing csv \n {csv}',
    input_variables = ['question','csv']
)

parser = StrOutputParser()

loader = CSVLoader(
    file_path="shortlist_2026-10-27.csv",
    encoding="utf-8"
)

data = loader.load()

print(data[0])

chain = prompt | model | parser

result = chain.invoke({'question':'What is this csv file about?','csv':data[0].page_content})
print(result)

