"""
agent.py
Construye el agente combinado de Mercado Central 24h:
- RAG sobre políticas (proveedores, atención al cliente, FAQ, reglamento interno)
- Consultas estructuradas sobre inventario (stock, precios, vencimientos, proveedores)

El agente usa "tool calling": el LLM decide cuándo y qué herramienta invocar
según la pregunta del usuario, y luego redacta una respuesta final en español.
"""
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src import config
from src.tools.policy_search import get_policy_retriever_tool
from src.tools.inventory_tools import ALL_INVENTORY_TOOLS

SYSTEM_PROMPT = """\
Eres el Asistente Virtual de Mercado Central 24h, una cadena de supermercados \
con operación en México y presencia en América Latina (Colombia, Chile, Perú, Argentina).

Tu trabajo es ayudar a empleados y clientes respondiendo preguntas sobre:
1. Políticas de la empresa: proveedores y compras, atención al cliente y \
devoluciones, preguntas frecuentes (FAQ) y reglamento interno.
2. Inventario de productos: disponibilidad, stock, precios, márgenes, \
proveedores y fechas de vencimiento.

Reglas importantes:
- Usa siempre las herramientas disponibles para buscar información real antes \
de responder; no inventes políticas ni datos de inventario.
- Si una pregunta mezcla política + inventario (ej. "¿puedo devolver este \
producto y cuánto stock queda?"), usa varias herramientas en la misma respuesta.
- Si la información solicitada no aparece en ninguna herramienta, dilo \
claramente en vez de inventar una respuesta.
- Responde siempre en español, de forma clara, concisa y profesional.
- Cuando cites una política, indica de qué documento proviene (por ejemplo: \
"según el Reglamento Interno...").
"""

# Nombres de herramientas -> etiqueta amigable para mostrar en la interfaz de chat
TOOL_LABELS = {
    "buscar_politicas_empresa": "Políticas",
    "buscar_producto": "Inventario",
    "consultar_stock_bajo": "Inventario",
    "productos_por_vencer": "Inventario",
    "info_proveedor": "Inventario",
    "calcular_margen": "Inventario",
}


def build_agent_executor() -> AgentExecutor:
    """Construye el AgentExecutor con el LLM configurado y todas las tools."""
    llm = _build_llm()

    tools = [get_policy_retriever_tool(), *ALL_INVENTORY_TOOLS]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=os.getenv("AGENT_VERBOSE", "false").lower() == "true",
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True,
    )
    return executor


def extract_sources(intermediate_steps) -> list:
    """
    A partir de los `intermediate_steps` devueltos por el AgentExecutor,
    obtiene la lista de etiquetas amigables de las herramientas que se
    usaron para responder (por ejemplo: ["Políticas", "Inventario"]).
    """
    labels = []
    for action, _observation in intermediate_steps:
        tool_name = getattr(action, "tool", None)
        label = TOOL_LABELS.get(tool_name, tool_name)
        if label and label not in labels:
            labels.append(label)
    return labels


def _build_llm():
    """Instancia el LLM según LLM_PROVIDER ("openai" por defecto, o "anthropic")."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.2,
        )
    elif config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=0.2,
        )
    else:
        raise ValueError(
            f"LLM_PROVIDER desconocido: '{config.LLM_PROVIDER}'. Usa 'openai' u 'anthropic'."
        )
