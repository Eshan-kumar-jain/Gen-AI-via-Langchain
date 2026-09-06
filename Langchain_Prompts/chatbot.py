import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "").strip()
model = ChatOpenAI(api_key=api_key)
chathistory = [
    SystemMessage(content = "You are a helpful assistant")
]

while True:
    user_input = input('You: ')
    chathistory.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chathistory)#static prompt
    chathistory.append(AIMessage(content = result.content))
    print("AI: ", result.content)

print("Here is you chat history: \n",chathistory)