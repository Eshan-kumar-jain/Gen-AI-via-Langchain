from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1", temperature=0.7)

result = model.invoke("what is a transformer?")
print(result.content)