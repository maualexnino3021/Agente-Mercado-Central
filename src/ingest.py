"""
ingest.py
Construye (o reconstruye) el índice vectorial FAISS a partir de los
documentos de políticas ubicados en data/documents/.

Usa la API de embeddings de OpenAI (OPENAI_EMBEDDING_MODEL, por defecto
"text-embedding-3-small") en vez de un modelo local de HuggingFace, para
mantener el despliegue liviano y rápido de construir en Render.

Uso:
    python -m src.ingest
    python -m src.ingest --force   # fuerza reconstrucción aunque ya exista el índice
"""
import argparse
import json
import sys

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

STATS_PATH_NAME = "stats.json"


def get_embeddings() -> OpenAIEmbeddings:
    if not config.OPENAI_API_KEY:
        raise EnvironmentError(
            "Falta OPENAI_API_KEY en el entorno. Es requerida para generar los "
            "embeddings del índice, incluso si el LLM de chat es Anthropic."
        )
    return OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


def load_documents() -> list:
    """Carga todos los .txt de data/documents/ como Documentos de LangChain."""
    docs = []
    txt_files = sorted(config.DOCUMENTS_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt en {config.DOCUMENTS_DIR}. "
            "Coloca ahí las políticas (manual de proveedores, atención al "
            "cliente, FAQ, reglamento interno, etc.)."
        )
    for path in txt_files:
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            # Metadato útil para citar la fuente en las respuestas del agente
            doc.metadata["source"] = path.stem
        docs.extend(loaded)
        print(f"  cargado: {path.name} ({len(loaded[0].page_content)} caracteres)")
    return docs


def build_vectorstore(force: bool = False) -> FAISS:
    """Construye el índice FAISS y lo persiste en disco."""
    index_file = config.VECTORSTORE_DIR / "index.faiss"
    embeddings = get_embeddings()

    if index_file.exists() and not force:
        print(f"Índice existente encontrado en {config.VECTORSTORE_DIR}. Usa --force para reconstruir.")
        return FAISS.load_local(
            str(config.VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
        )

    print("Cargando documentos de políticas...")
    documents = load_documents()

    print("Dividiendo documentos en fragmentos (chunks)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(documents)} documentos -> {len(chunks)} fragmentos")

    print(f"Generando embeddings con '{config.OPENAI_EMBEDDING_MODEL}' (API de OpenAI) e indexando en FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.VECTORSTORE_DIR))

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": config.OPENAI_EMBEDDING_MODEL,
    }
    stats_path = config.VECTORSTORE_DIR / STATS_PATH_NAME
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Índice guardado en {config.VECTORSTORE_DIR}")

    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Construye el índice vectorial de políticas.")
    parser.add_argument("--force", action="store_true", help="Reconstruir el índice aunque ya exista.")
    args = parser.parse_args()

    try:
        build_vectorstore(force=args.force)
    except Exception as exc:
        print(f"ERROR durante la ingesta: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
