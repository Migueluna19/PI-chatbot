# Embeddings y modelo de chat de OpenAI.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Base vectorial Chroma.
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings()

persist_dir = "chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir,
)

print(f"Vector store creado. Persistencia en: {persist_dir}")