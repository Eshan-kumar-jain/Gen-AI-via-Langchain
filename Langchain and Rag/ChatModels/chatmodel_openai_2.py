from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY", "").strip()

temps = [0, 0.3, 0.7, 1.0, 1.3, 1.7]

for t in temps:
    model = ChatOpenAI(
        model="gpt-4",
        temperature=t,
        max_tokens=100,
        api_key=api_key,
    )
    result = model.invoke("Suggest me a 4 line poem on football")
    print(f"temperature={t}:\n{result.content}\n")