import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv(override=True)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=300,
    api_key=os.getenv("OPENAI_API_KEY", "").strip(),
)

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about Ms dhoni"

docs_embedding = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],docs_embedding)[0] # both in 2d list

index, score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1] ## sort on the bases of the similarity scores , enumerate gives the index to the scores as well

print(query)
print(docs[index])
print("similarity score is:", score)