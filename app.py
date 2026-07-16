"""
app.py
Servidor Flask para el Asistente Virtual de Mercado Central 24h.

Expone:
- GET  /             -> interfaz de chat (templates/index.html)
- POST /api/chat      -> recibe {"message": "..."} y devuelve
                         {"answer": "...", "sources": ["Políticas", "Inventario"]}
- GET  /api/health    -> healthcheck para Render: {"status": "ok", ...}

El agente (LangChain AgentExecutor con tool-calling) se construye una sola
vez al arrancar el proceso, no en cada request, para evitar reconstruir las
herramientas (y volver a cargar el índice FAISS) en cada pregunta.
"""
import json
import logging

from flask import Flask, jsonify, render_template, request

from src import config
from src.agent import build_agent_executor, extract_sources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agente-mercado-central")

app = Flask(__name__)

_executor = None
_init_error = None


def get_executor():
    """Construye (una sola vez) y devuelve el AgentExecutor. Cachea errores
    de inicialización para no reintentar costosamente en cada request."""
    global _executor, _init_error
    if _executor is None and _init_error is None:
        try:
            config.validate_config()
            _executor = build_agent_executor()
            logger.info(
                "Agente inicializado correctamente (LLM_PROVIDER=%s)",
                config.LLM_PROVIDER,
            )
        except Exception as exc:  # noqa: BLE001
            _init_error = str(exc)
            logger.error("Error al inicializar el agente: %s", _init_error)
    return _executor, _init_error


def _read_index_stats() -> dict:
    """Lee data/vectorstore/stats.json (generado por src/ingest.py) si existe."""
    stats_path = config.VECTORSTORE_DIR / "stats.json"
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# Se inicializa al importar el módulo (cuando gunicorn arranca el worker),
# para que el primer request real no pague el costo de arranque.
get_executor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    executor, error = get_executor()
    if error:
        return jsonify({"status": "error", "detail": error}), 500

    stats = _read_index_stats()
    return jsonify({
        "status": "ok",
        "provider": config.LLM_PROVIDER,
        "chat_model": config.OPENAI_MODEL if config.LLM_PROVIDER == "openai" else config.ANTHROPIC_MODEL,
        "embedding_model": config.OPENAI_EMBEDDING_MODEL,
        "documents": stats.get("documents"),
        "chunks": stats.get("chunks"),
        "tools": [t.name for t in executor.tools],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    executor, error = get_executor()
    if error:
        return jsonify({"error": f"El agente no está disponible: {error}"}), 500

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "El campo 'message' es requerido."}), 400

    try:
        result = executor.invoke({"input": message, "chat_history": []})
        answer = result.get("output", "")
        sources = extract_sources(result.get("intermediate_steps", []))
        return jsonify({"answer": answer, "sources": sources})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error al procesar la pregunta")
        return jsonify({"error": f"Ocurrió un error al procesar tu pregunta: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
