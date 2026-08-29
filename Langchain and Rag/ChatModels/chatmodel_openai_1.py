from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

model = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY", "").strip()
)
result = model.invoke("What is the meaning of Akhand Bharat?")

print(result.content)