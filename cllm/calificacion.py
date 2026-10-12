"""Calificación de las prácticas.

Cada prueba vale un punto. `calificar` ejecuta la prueba, guarda el
resultado y lo imprime. `resumen` muestra el total al final del notebook.
"""

puntos = {}


def calificar(numero, prueba, *args, **kwargs):
    """Ejecuta `prueba` y asigna 1 punto si no lanza ninguna excepción."""
    try:
        prueba(*args, **kwargs)
        puntos[numero] = 1
        print(f"✅ Punto {numero}: correcto. (+1)")
    except AssertionError as e:
        puntos[numero] = 0
        print(f"❌ Punto {numero}: la prueba falló. {e}")
    except Exception as e:
        puntos[numero] = 0
        print(f"❌ Punto {numero}: error al ejecutar tu código -> {type(e).__name__}: {e}")


def resumen(total):
    """Imprime la calificación de los puntos 1..`total`."""
    obtenidos = sum(puntos.get(n, 0) for n in range(1, total + 1))
    for n in range(1, total + 1):
        print(f"Punto {n}: {'✅' if puntos.get(n) else '❌'}")
    print(f"\nPuntos obtenidos: {obtenidos} / {total}")
    return obtenidos
