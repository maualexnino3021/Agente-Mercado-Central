"""
tools/inventory_tools.py
Tools estructuradas que consultan el inventario de productos
(data/inventario_mercado_central.xlsx) usando pandas.

A diferencia de policy_search.py (RAG semántico sobre texto libre),
estas tools hacen consultas exactas/filtradas sobre datos tabulares:
stock, precios, vencimientos, proveedores, etc.
"""
from datetime import datetime
from typing import Optional

import pandas as pd
from langchain.tools import tool

from src import config

_df_cache: Optional[pd.DataFrame] = None


def _load_inventory() -> pd.DataFrame:
    """Carga (con cache en memoria) el inventario desde el Excel."""
    global _df_cache
    if _df_cache is None:
        df = pd.read_excel(config.INVENTORY_PATH)
        df.columns = [c.strip() for c in df.columns]
        for date_col in ("Fecha de Fabricación", "Fecha de Vencimiento"):
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        _df_cache = df
    return _df_cache


def reload_inventory() -> None:
    """Fuerza releer el Excel en el próximo acceso (útil tras actualizar el archivo)."""
    global _df_cache
    _df_cache = None


def _format_rows(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df.empty:
        return "No se encontraron resultados."
    cols = [
        "SKU", "Descripción", "Marca", "Categoría", "Stock Actual",
        "Stock Mínimo", "Precio de Venta Unitario", "Proveedor Principal",
        "Fecha de Vencimiento",
    ]
    cols = [c for c in cols if c in df.columns]
    shown = df[cols].head(max_rows)
    lines = []
    for _, row in shown.iterrows():
        parts = [f"{c}: {row[c]}" for c in cols]
        lines.append(" | ".join(str(p) for p in parts))
    text = "\n".join(lines)
    if len(df) > max_rows:
        text += f"\n... y {len(df) - max_rows} resultado(s) más (refina la búsqueda si necesitas verlos)."
    return text


@tool
def buscar_producto(nombre_o_sku: str) -> str:
    """
    Busca productos en el inventario por SKU exacto o por coincidencia parcial
    en la descripción, marca o categoría (búsqueda insensible a mayúsculas/acentos simples).
    Devuelve stock actual, precio, proveedor y fecha de vencimiento.
    Útil para responder preguntas como "¿tienen arroz integral?" o "stock del SKU MER-005".
    """
    df = _load_inventory()
    query = nombre_o_sku.strip()

    exact = df[df["SKU"].astype(str).str.lower() == query.lower()]
    if not exact.empty:
        return _format_rows(exact)

    mask = (
        df["Descripción"].astype(str).str.contains(query, case=False, na=False)
        | df["Marca"].astype(str).str.contains(query, case=False, na=False)
        | df["Categoría"].astype(str).str.contains(query, case=False, na=False)
        | df["Subcategoría"].astype(str).str.contains(query, case=False, na=False)
    )
    result = df[mask]
    return _format_rows(result)


@tool
def consultar_stock_bajo() -> str:
    """
    Lista todos los productos cuyo stock actual está en o por debajo de su
    stock mínimo definido (necesitan reabastecimiento urgente).
    Útil para preguntas como "¿qué productos hay que reponer?" o
    "¿qué está por agotarse?".
    """
    df = _load_inventory()
    result = df[df["Stock Actual"] <= df["Stock Mínimo"]]
    return _format_rows(result, max_rows=25)


@tool
def productos_por_vencer(dias: int = 30) -> str:
    """
    Lista los productos cuya Fecha de Vencimiento cae dentro de los próximos
    N días (por defecto 30). Útil para preguntas como
    "¿qué productos vencen pronto?" o "productos a punto de caducar".

    Args:
        dias: número de días hacia adelante a considerar (por defecto 30).
    """
    df = _load_inventory()
    hoy = pd.Timestamp(datetime.now().date())
    limite = hoy + pd.Timedelta(days=dias)
    mask = (df["Fecha de Vencimiento"] >= hoy) & (df["Fecha de Vencimiento"] <= limite)
    result = df[mask].sort_values("Fecha de Vencimiento")
    return _format_rows(result, max_rows=25)


@tool
def info_proveedor(nombre_proveedor: str) -> str:
    """
    Lista todos los productos asociados a un proveedor específico
    (búsqueda parcial, insensible a mayúsculas), incluyendo tiempos de
    reposición. Útil para preguntas como
    "¿qué productos nos surte Distribuidora Granos S.A.?".
    """
    df = _load_inventory()
    mask = df["Proveedor Principal"].astype(str).str.contains(
        nombre_proveedor, case=False, na=False
    )
    result = df[mask]
    if result.empty:
        return "No se encontraron productos para ese proveedor."
    tiempos = result["Tiempo de Reposición"].dropna().unique()
    resumen = f"Proveedor(es) encontrados. Tiempo(s) de reposición: {', '.join(map(str, tiempos))}\n\n"
    return resumen + _format_rows(result, max_rows=20)


@tool
def calcular_margen(nombre_o_sku: str) -> str:
    """
    Calcula el margen de ganancia (precio de venta - costo unitario) y el
    porcentaje de margen para un producto dado (por SKU o nombre).
    Útil para preguntas como "¿cuál es el margen del arroz integral?".
    """
    df = _load_inventory()
    query = nombre_o_sku.strip()
    exact = df[df["SKU"].astype(str).str.lower() == query.lower()]
    result = exact if not exact.empty else df[
        df["Descripción"].astype(str).str.contains(query, case=False, na=False)
    ]
    if result.empty:
        return "No se encontró el producto."

    lines = []
    for _, row in result.iterrows():
        costo = row["Costo Unitario"]
        precio = row["Precio de Venta Unitario"]
        margen = precio - costo
        pct = (margen / precio * 100) if precio else 0
        lines.append(
            f"{row['SKU']} - {row['Descripción']} ({row['Marca']}): "
            f"costo ${costo:.2f}, precio ${precio:.2f}, "
            f"margen ${margen:.2f} ({pct:.1f}%)"
        )
    return "\n".join(lines)


ALL_INVENTORY_TOOLS = [
    buscar_producto,
    consultar_stock_bajo,
    productos_por_vencer,
    info_proveedor,
    calcular_margen,
]
