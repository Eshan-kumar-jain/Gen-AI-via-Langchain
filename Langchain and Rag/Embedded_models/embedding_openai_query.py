import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32,
    api_key=os.getenv("OPENAI_API_KEY", "").strip(),
)

result = embedding.embed_query("What is the recipe of samosa?")

print(len(result))
print(result)
print(str(result))