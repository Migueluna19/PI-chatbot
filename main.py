import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.build_text_loader import load_documents_from_paths

load_dotenv()

PERSIST_DIR = "chroma_db"
DOCUMENT_PATHS = [Path("data/faq_document.txt")]


def indexar_documentos() -> Chroma:
    """Pasos 1-3: cargar, dividir y embeber. Solo se ejecuta la primera vez."""

    # Paso 1 — text_loader: cargar documentos desde archivo(s)
    print("\n=== Paso 1: Cargando documentos ===")
    documents = load_documents_from_paths(DOCUMENT_PATHS)

    # Paso 2 — build_text_chunks: dividir documentos en chunks
    print("\n=== Paso 2: Dividiendo en chunks ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Número de chunks: {len(chunks)}")

    # Paso 3 — build_text_embeddings: crear embeddings y persistir en Chroma
    print("\n=== Paso 3: Creando vector store ===")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Vector store creado y persistido en: {PERSIST_DIR}")
    return vectorstore


def build_retriever(vectorstore: Chroma, search_type: str, all_chunks: list | None = None):
    """Construye el retriever según el método de búsqueda vectorial seleccionado.

    - similarity  → k-NN: los k vecinos más cercanos por similitud coseno
    - mmr         → ANN/MMR: maximiza relevancia y penaliza redundancia entre resultados
    - threshold   → Rango: filtra chunks bajo un umbral mínimo de similitud
    - hybrid      → Híbrido: combina BM25 (léxico) con vectorial (semántico)
    """
    if search_type == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

    elif search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
        )

    elif search_type == "threshold":
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.7},
        )

    elif search_type == "hybrid":
        if not all_chunks:
            raise ValueError("Se requieren los chunks cargados para BM25 en modo híbrido.")
        bm25_retriever = BM25Retriever.from_documents(all_chunks, k=3)
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )

    else:
        raise ValueError(f"search_type no reconocido: {search_type!r}")


def build_json_response(query: str, result: dict) -> dict:
    """Formatea la respuesta del pipeline como JSON estructurado para transparencia y auditabilidad."""
    chunks_related = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "desconocido"),
            "page": doc.metadata.get("page", None),
        }
        for doc in result.get("source_documents", [])
    ]
    return {
        "user_question": query,
        "system_answer": result["result"],
        "chunks_related": chunks_related,
    }


def evaluate_response(query: str, answer: str, chunks: list[dict], llm: ChatOpenAI) -> dict:
    """Agente evaluador: puntúa la calidad de la respuesta RAG en tres dimensiones (0-10).

    Dimensiones evaluadas:
    - relevancia_chunks: qué tan pertinentes son los chunks recuperados para la pregunta
    - precision:         qué tan exacta y fundamentada en los chunks es la respuesta
    - completitud:       qué tan completa es la respuesta respecto a lo que pregunta el usuario

    Devuelve un dict con score, justificación por dimensión y puntuación global promedio.
    """
    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}] {c['content']}" for i, c in enumerate(chunks)
    )

    prompt = f"""Eres un evaluador experto de sistemas RAG (Retrieval-Augmented Generation).
Analiza la siguiente interacción y puntúa cada dimensión del 0 al 10.

### Pregunta del usuario
{query}

### Chunks recuperados del vector store
{chunks_text}

### Respuesta generada por el LLM
{answer}

### Instrucciones de evaluación
Evalúa estas tres dimensiones:

1. **relevancia_chunks** (0-10): ¿Los chunks recuperados son pertinentes para responder la pregunta?
   - 0 = completamente irrelevantes
   - 10 = perfectamente alineados con la pregunta

2. **precision** (0-10): ¿La respuesta es exacta y está fundamentada en los chunks?
   - 0 = la respuesta contradice o ignora los chunks
   - 10 = la respuesta refleja fielmente lo que dicen los chunks

3. **completitud** (0-10): ¿La respuesta cubre todos los aspectos de la pregunta?
   - 0 = respuesta vacía o evasiva
   - 10 = respuesta exhaustiva que no deja aspectos sin atender

Responde ÚNICAMENTE con un JSON válido con esta estructura exacta:
{{
  "relevancia_chunks": {{"score": <0-10>, "justificacion": "<una oración>"}},
  "precision": {{"score": <0-10>, "justificacion": "<una oración>"}},
  "completitud": {{"score": <0-10>, "justificacion": "<una oración>"}},
  "puntuacion_global": <promedio de los tres scores, un decimal>
}}"""

    evaluator_llm = ChatOpenAI(
        model=llm.model_name,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    raw = evaluator_llm.invoke(prompt)
    return json.loads(raw.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot RAG — consulta documentos con IA")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Pregunta para el LLM (si se omite, se pedirá de forma interactiva)",
    )
    parser.add_argument(
        "--search-type",
        choices=["similarity", "mmr", "threshold", "hybrid"],
        default="similarity",
        help=(
            "Método de búsqueda vectorial: "
            "similarity (k-NN), mmr (ANN/diversidad), "
            "threshold (rango), hybrid (BM25+vector). "
            "Default: similarity"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings = OpenAIEmbeddings()

    # Cargar DB existente o indexar por primera vez
    if os.path.exists(PERSIST_DIR):
        print(f"Vector store encontrado. Cargando desde: {PERSIST_DIR}")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
    else:
        vectorstore = indexar_documentos()

    # Para búsqueda híbrida se necesitan los chunks en memoria para BM25
    all_chunks = None
    if args.search_type == "hybrid":
        documents = load_documents_from_paths(DOCUMENT_PATHS)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = text_splitter.split_documents(documents)

    # build_similarity_chunks: construir retriever según método seleccionado
    print(f"\n=== Método de búsqueda: {args.search_type} ===")
    retriever = build_retriever(vectorstore, args.search_type, all_chunks)

    # Configurar LLM y cadena RAG
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    print("\nCadena RAG lista.")

    # Consulta — desde argumento CLI o de forma interactiva
    query = args.query or input("\nIngresa tu pregunta: ").strip()
    result = qa_chain.invoke({"query": query})

    # Salida estructurada en JSON con user_question, system_answer y chunks_related
    response = build_json_response(query, result)

    # Agente evaluador: puntúa calidad de la respuesta en relevancia, precisión y completitud
    print("\n=== Evaluando calidad de la respuesta ===")
    response["evaluation"] = evaluate_response(
        query=response["user_question"],
        answer=response["system_answer"],
        chunks=response["chunks_related"],
        llm=llm,
    )

    print("\n" + json.dumps(response, ensure_ascii=False, indent=2))

    # Persistir respuesta en outputs/sample_queries.json (acumula todas las consultas)
    output_path = Path("outputs/sample_queries.json")
    output_path.parent.mkdir(exist_ok=True)

    history = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
    history.append(response)
    output_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRespuesta guardada en: {output_path}")


if __name__ == "__main__":
    main()
