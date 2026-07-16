"""
config.py
Configuración central del proyecto: rutas, modelos y parámetros del agente.
Todos los valores sensibles (API keys) se leen desde variables de entorno (.env).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
INVENTORY_PATH = DATA_DIR / "inventario_mercado_central.xlsx"

# --------------------------------------------------------------------------
# Proveedor de LLM: "openai" (por defecto) u "anthropic"
# --------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Modelos por defecto (pueden sobreescribirse en .env)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# API Keys (se leen de entorno; NUNCA se hardcodean)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --------------------------------------------------------------------------
# Embeddings para el vectorstore (RAG sobre políticas)
#
# Se usa la API de embeddings de OpenAI (en vez de un modelo local de
# HuggingFace) para mantener el despliegue liviano: evita instalar
# `sentence-transformers`/`torch` (cientos de MB) y descargar un modelo
# al arrancar, lo cual es un riesgo real de timeout/memoria en el plan
# gratuito de Render. Por eso, la clave de OpenAI es SIEMPRE requerida,
# incluso si el LLM de chat usado es Anthropic (ver validate_config()).
# --------------------------------------------------------------------------
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --------------------------------------------------------------------------
# Parámetros de chunking / retrieval
# --------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))

# --------------------------------------------------------------------------
# Parámetros de negocio (inventario)
# --------------------------------------------------------------------------
DIAS_ALERTA_VENCIMIENTO = int(os.getenv("DIAS_ALERTA_VENCIMIENTO", "30"))

# --------------------------------------------------------------------------
# Validaciones básicas
# --------------------------------------------------------------------------
def validate_config() -> None:
    """Lanza un error claro si falta configuración crítica."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "Falta OPENAI_API_KEY en el entorno (.env). Se requiere siempre, "
            "incluso si LLM_PROVIDER='anthropic', porque los embeddings del "
            "RAG usan la API de OpenAI. Consigue una clave en "
            "https://platform.openai.com/api-keys"
        )
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "LLM_PROVIDER='anthropic' pero falta ANTHROPIC_API_KEY en el entorno (.env). "
            "Consigue una clave en https://console.anthropic.com/"
        )
    if LLM_PROVIDER not in ("openai", "anthropic"):
        raise ValueError(
            f"LLM_PROVIDER desconocido: '{LLM_PROVIDER}'. Usa 'openai' u 'anthropic'."
        )
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de inventario: {INVENTORY_PATH}")
    if not DOCUMENTS_DIR.exists() or not any(DOCUMENTS_DIR.iterdir()):
        raise FileNotFoundError(
            f"No se encontraron documentos de políticas en: {DOCUMENTS_DIR}"
        )
