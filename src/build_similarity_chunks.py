from langchain_classic.chains import RetrievalQA

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Vista previa opcional: qué chunks recuperaría una consulta.
preview_query = "¿Cuales son los comercios participantes?"
preview_docs = retriever.invoke(preview_query)
print(f"Preview retrieval para: {preview_query!r}")
for i, doc in enumerate(preview_docs, start=1):
    print(f"\n[{i}] {doc.page_content[:800]}...")