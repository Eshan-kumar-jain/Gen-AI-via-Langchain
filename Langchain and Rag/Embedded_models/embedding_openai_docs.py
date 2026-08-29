import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32,
    api_key=os.getenv("OPENAI_API_KEY", "").strip(),
)

docs = [
    "How do you make samosas?",
    "Recipe for potato-filled fried pastry",
    "The stock market closed higher today",
]

result = embedding.embed_documents(docs)

print(len(result))
print(result)
print(str(result))