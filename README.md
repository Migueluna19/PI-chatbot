# PI Chatbot — RAG con LangChain y OpenAI

Chatbot de preguntas y respuestas sobre documentos propios usando **Retrieval-Augmented Generation (RAG)**, construido con LangChain, Chroma y la API de OpenAI.

---

## Requisitos previos

- Python 3.14 o superior
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado en tu sistema
- Una API Key de OpenAI

---

## Instalación

### 1. Clona el repositorio

```bash
git clone <url-del-repositorio>
cd PI_chatbot
```

### 2. Inicializa el proyecto con uv

```bash
uv init
```

### 3. Sincroniza el entorno virtual

```bash
uv sync
```

### 4. Instala las dependencias desde requirements.txt

```bash
uv add -r requirements.txt
```

### 5. Configura tu API Key de OpenAI

Copia el archivo de ejemplo y agrega tu clave:

```bash
cp .env.example .env
```

Abre `.env` y reemplaza el valor vacío con tu API Key:

```
OPENAI_API_KEY='sk-...'
```

---

## Uso

### Con la pregunta directamente en la terminal

```bash
uv run python main.py "¿Cuándo es válida la promoción?"
```

### Sin argumento (modo interactivo)

Si no pasas la pregunta, el programa te la pedirá al ejecutarse:

```bash
uv run python main.py
```

```
Ingresa tu pregunta: ¿Cuándo es válida la promoción?
```

### Elegir el método de búsqueda vectorial (`--search-type`)

El flag `--search-type` controla cómo se recuperan los chunks relevantes antes de pasarlos al LLM:

| Valor | Método | Descripción |
|-------|--------|-------------|
| `similarity` | k-NN | Los k vecinos más cercanos por similitud coseno *(default)* |
| `mmr` | ANN/MMR | Maximiza relevancia y penaliza chunks redundantes entre sí |
| `threshold` | Rango | Solo retorna chunks con similitud ≥ 0.7 |
| `hybrid` | Híbrido | Combina BM25 (léxico) con vectorial (semántico) en proporción 50/50 |

```bash
# k-NN (default)
uv run python main.py "¿Cuáles son los comercios participantes?"

# ANN/MMR — resultados diversos, sin repetición
uv run python main.py --search-type mmr "¿Qué beneficios ofrece?"

# Rango — solo chunks con alta similitud
uv run python main.py --search-type threshold "¿Cuándo vence la promoción?"

# Híbrido BM25 + vectorial
uv run python main.py --search-type hybrid "¿Cuál es el proceso de registro?"

# Modo interactivo con método específico
uv run python main.py --search-type hybrid
```

### Salida JSON estructurada

Cada consulta devuelve un JSON con tres campos para garantizar transparencia y auditabilidad:

```json
{
  "user_question": "¿Cuáles son los comercios participantes?",
  "system_answer": "Los comercios participantes son ...",
  "chunks_related": [
    {
      "content": "Texto del fragmento recuperado del documento...",
      "source": "data/faq_document.txt",
      "page": null
    }
  ]
}
```

| Campo | Descripción |
|-------|-------------|
| `user_question` | La pregunta original del usuario |
| `system_answer` | La respuesta generada por el LLM |
| `chunks_related` | Los fragmentos del documento usados como contexto, con su fuente y página |

### Ver ayuda y opciones disponibles

```bash
uv run python main.py --help
```

---

## Estructura del proyecto

```
PI_chatbot/
├── main.py               # Punto de entrada principal — cadena RAG completa
├── text_loader.py        # Paso 1: carga de documentos (.pdf, .txt, .md)
├── text_chunks.py        # Paso 2: división de documentos en chunks
├── text_embeddings.py    # Paso 3: generación de embeddings y persistencia en Chroma
├── similarity_chunks.py  # Paso 4: recuperación por similaridad
├── chroma_db/            # Vector store persistido (se crea al primer uso)
├── requirements.txt      # Dependencias del proyecto
├── pyproject.toml        # Configuración de uv
├── .env.example          # Plantilla de variables de entorno
└── .env                  # Variables de entorno (no se sube al repositorio)
```

---

## Notas

- La carpeta `chroma_db/` se genera automáticamente la primera vez que se ejecuta `main.py`. Las siguientes ejecuciones reutilizan el vector store ya creado.
- Para cambiar el documento fuente, edita la variable `DOCUMENT_PATHS` en `main.py`.
