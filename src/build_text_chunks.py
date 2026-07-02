
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)
print(f"Número de chunks: {len(chunks)}")
for i in range(len(chunks)):
    print(chunks[i].page_content, "...")
    print("*"*100)