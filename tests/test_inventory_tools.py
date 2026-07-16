"""
tests/test_inventory_tools.py
Pruebas unitarias de las tools de inventario. No requieren API key
porque no llaman al LLM: prueban directamente la lógica de pandas.

Ejecutar con:
    pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools import inventory_tools as it


def test_buscar_producto_por_sku():
    resultado = it.buscar_producto.invoke({"nombre_o_sku": "MER-001"})
    assert "MER-001" in resultado
    assert "Arroz Blanco" in resultado


def test_buscar_producto_por_nombre_parcial():
    resultado = it.buscar_producto.invoke({"nombre_o_sku": "arroz"})
    assert "Arroz" in resultado or "arroz" in resultado.lower()


def test_buscar_producto_inexistente():
    resultado = it.buscar_producto.invoke({"nombre_o_sku": "PRODUCTO-QUE-NO-EXISTE-XYZ"})
    assert "No se encontraron" in resultado


def test_consultar_stock_bajo_no_lanza_error():
    resultado = it.consultar_stock_bajo.invoke({})
    assert isinstance(resultado, str)


def test_productos_por_vencer_no_lanza_error():
    resultado = it.productos_por_vencer.invoke({"dias": 60})
    assert isinstance(resultado, str)


def test_info_proveedor():
    resultado = it.info_proveedor.invoke({"nombre_proveedor": "Distribuidora Granos"})
    assert "Distribuidora" in resultado or "No se encontraron" in resultado


def test_calcular_margen():
    resultado = it.calcular_margen.invoke({"nombre_o_sku": "MER-001"})
    assert "margen" in resultado.lower()
