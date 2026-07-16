"""
main.py
Punto de entrada CLI: chat interactivo por consola con el agente de
Mercado Central 24h.

Uso:
    python -m src.main
"""
import sys

from src import config
from src.agent import build_agent_executor


BANNER = """\
================================================================
  Asistente Mercado Central 24h  (políticas + inventario)
  Proveedor de LLM: {provider}
  Escribe tu pregunta, o 'salir' para terminar.
================================================================
"""


def main():
    try:
        config.validate_config()
    except Exception as exc:
        print(f"Error de configuración: {exc}")
        sys.exit(1)

    print(BANNER.format(provider=config.LLM_PROVIDER))

    try:
        executor = build_agent_executor()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    chat_history = []

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta luego!")
            break

        result = executor.invoke({"input": user_input, "chat_history": chat_history})
        answer = result["output"]
        print(f"\nAgente: {answer}\n")

        chat_history.append(("human", user_input))
        chat_history.append(("ai", answer))


if __name__ == "__main__":
    main()
