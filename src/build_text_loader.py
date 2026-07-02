# Document loaders: TextLoader lee .txt; PyPDFLoader extrae texto de PDFs.
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# División en chunks. En LangChain 1.x vive en langchain_text_splitters
# (en material antiguo: from langchain.text_splitter import RecursiveCharacterTextSplitter).
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from pathlib import Path

def load_documents_from_path(path: Path) -> list:
    """Routing por extensión: .pdf → PyPDFLoader, .txt/.md → TextLoader."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Extensión no soportada: {suffix!r}. Usa .pdf, .txt o .md."
        )

    return loader.load()


def load_documents_from_paths(paths: list[Path]) -> list:
    documents = []
    for path in paths:
        loaded = load_documents_from_path(path)
        documents.extend(loaded)
        print(
            f"  {path.name} ({path.suffix.lower()}) → "
            f"{len(loaded)} documento(s) LangChain"
        )
    return documents


# --- Ejecución directa (python build_text_loader.py) ---
if __name__ == "__main__":
    document_path = Path("data/faq_document.txt")
    paths_to_load = [document_path]

    print("Cargando:")
    documents = load_documents_from_paths(paths_to_load)

    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"\nTotal objetos Document: {len(documents)}")
    print(f"Caracteres totales (aprox.): {total_chars}")
    if documents:
        preview = documents[0].page_content[:200].replace("\n", " ")
        print(f"Preview: {preview}...")