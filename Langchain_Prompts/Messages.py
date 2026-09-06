import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "").strip()
model = ChatOpenAI(api_key=api_key)

messages =[
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = "Tell me about langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)