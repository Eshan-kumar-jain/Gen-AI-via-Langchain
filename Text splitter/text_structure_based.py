from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """ 
My Name is Eshan
I am 26 years old

I live in Mumbai
How are you
"""
chunk = input("Enter the chunk size: ")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = int(chunk),
    chunk_overlap = 0  
)


chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)