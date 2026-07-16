"""
tools/policy_search.py
Tool de recuperación (RAG) sobre las políticas de la empresa:
- Manual de Proveedores y Política de Compras
- Política de Atención al Cliente, Cambios y Devoluciones
- Preguntas Frecuentes (FAQ)
- Reglamento Interno y Procedimientos Operativos

El agente usa esta tool para responder preguntas sobre políticas,
procedimientos, condiciones comerciales, reglas internas, etc.

Usa la API de embeddings de OpenAI (ver src/ingest.py) para cargar el
mismo índice FAISS que se construyó durante el despliegue.
"""
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

from src import config


def get_policy_retriever_tool():
    """
    Carga el índice FAISS ya construido (ver src/ingest.py) y lo expone
    como una tool de LangChain lista para usar en un agente con tool-calling.
    """
    if not (config.VECTORSTORE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "No existe el índice vectorial. Ejecuta primero:\n"
            "    python -m src.ingest"
        )
    if not config.OPENAI_API_KEY:
        raise EnvironmentError(
            "Falta OPENAI_API_KEY en el entorno. Es requerida para generar los "
            "embeddings de cada consulta, incluso si el LLM de chat es Anthropic."
        )

    embeddings = OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    vectorstore = FAISS.load_local(
        str(config.VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})

    tool = create_retriever_tool(
        retriever,
        name="buscar_politicas_empresa",
        description=(
            "Busca información en las políticas oficiales de Mercado Central 24h: "
            "manual de proveedores y compras, política de atención al cliente y "
            "devoluciones, preguntas frecuentes (FAQ) para clientes y empleados, "
            "y el reglamento interno de la empresa. "
            "Úsala para cualquier pregunta sobre procedimientos, condiciones "
            "comerciales, reglas de devolución, beneficios de empleados, "
            "horarios, políticas de calidad, ética, etc. "
            "El argumento de entrada debe ser la pregunta o tema a buscar, "
            "en español y lo más específico posible."
        ),
    )
    return tool
