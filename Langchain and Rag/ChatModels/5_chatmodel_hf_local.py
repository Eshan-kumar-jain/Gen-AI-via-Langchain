from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    device=0,
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 100,
        "do_sample": True,
    },
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Is Balochistan an independent nation?")
print(result.content)