from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Bhopal is the ca[ital of MP"

docs = ["Delhi is the capital of india",
       "Punjab is a state in india",
       "hyderbad is the cappital of telangana"]

vector = embedding.embed_documents(docs)

print(str(vector))