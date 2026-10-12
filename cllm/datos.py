"""Textos de ejemplo usados por las prácticas."""

from pathlib import Path

_EN_PAZ = Path(__file__).resolve().parent.parent / "en_paz.txt"


def cargar_en_paz(ruta=None):
    """Lee el poema *En paz* de Amado Nervo y regresa el texto."""
    ruta = _EN_PAZ if ruta is None else Path(ruta)
    return ruta.read_text(encoding="utf-8")
