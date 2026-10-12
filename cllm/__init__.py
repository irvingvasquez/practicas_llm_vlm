"""Utilidades compartidas del curso de LLMs.

Los notebooks importan de aquí la calificación y los datos de ejemplo,
para que cada práctica sólo contenga el ejercicio.
"""

from cllm.calificacion import calificar, resumen
from cllm.datos import cargar_en_paz

__all__ = ["calificar", "resumen", "cargar_en_paz"]
