# Prácticas de PyTorch: construyendo un LLM desde cero

Prácticas en Jupyter Notebook para aprender PyTorch y, a lo largo de varias lecciones, construir y entrenar un modelo de lenguaje pequeño basado en GPT-2.

Cada notebook contiene secciones marcadas con `TODO` que debes completar. Después de cada una hay una celda de **prueba** que verifica tu código y, si es correcto, te asigna el punto. Al final del notebook verás tu calificación.

## Prácticas

| Notebook | Tema | Puntos |
|---|---|---|
| `02_01_preprocesamiento_texto_llm.ipynb` | Tokenización, BPE, *data loaders* y embeddings para un LLM | 6 |
| `03_01_mecanismos_atencion.ipynb` | Self-attention, atención causal y multi-cabeza | 6 |

La calificación y las utilidades compartidas viven en el paquete `cllm` (curso LLM), junto a los notebooks. Así cada práctica sólo contiene el ejercicio. No hace falta instalarlo: Jupyter lo encuentra si abres el notebook desde esta carpeta.

```python
from cllm import calificar, resumen, cargar_en_paz
from cllm.pruebas_02_01 import prueba_1, prueba_2, prueba_3, prueba_4, prueba_5, prueba_6
```

## Opción 1: Google Colab (sin instalar nada)

Abre el notebook en GitHub y haz clic en el botón **Open in Colab** al inicio. Colab ya incluye PyTorch; la primera celda de cada notebook instala lo que haga falta.

Guarda tu trabajo con **Archivo → Guardar una copia en Drive**; de lo contrario perderás tus cambios al cerrar la pestaña.

## Opción 2: Ejecutar en tu computadora con conda

### 1. Instala conda

Si no tienes conda, instala [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (o Anaconda). Verifica la instalación abriendo una terminal (en Windows, **Anaconda Prompt**):

```bash
conda --version
```

### 2. Clona el repositorio

```bash
git clone https://github.com/irvingvasquez/practicas_llm_vlm.git
cd practicas_llm_vlm
```

Si no tienes `git`, descarga el repositorio como ZIP desde el botón **Code → Download ZIP** y descomprímelo.

### 3. Crea y activa el entorno

```bash
conda create -n llm python=3.11 -y
conda activate llm
```

Deberás ejecutar `conda activate llm` cada vez que abras una nueva terminal para trabajar en las prácticas.

### 4. Instala PyTorch

**Sólo CPU** (funciona en cualquier computadora, suficiente para estas prácticas):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**macOS** (incluye aceleración con chips Apple Silicon):

```bash
pip install torch
```

**Con GPU NVIDIA**: entra a [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), selecciona tu sistema operativo, *Pip* y tu versión de CUDA, y copia el comando que te muestra. Por ejemplo:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> Desde la versión 2.6, PyTorch ya no se publica en el canal de conda; por eso se instala con `pip` dentro del entorno de conda.

### 5. Instala las demás dependencias

```bash
pip install tiktoken jupyterlab matplotlib numpy
```

### 6. Verifica la instalación

```bash
python -c "import torch, tiktoken; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.is_available())"
```

Debe imprimir la versión de PyTorch sin errores. `GPU: False` es normal si no tienes una tarjeta NVIDIA.

### 7. Abre los notebooks

```bash
jupyter lab
```

Se abrirá el navegador. Abre el notebook de la práctica y ejecuta las celdas en orden con **Shift + Enter**.

La celda `!pip install -q tiktoken` al inicio de algunos notebooks no hace falta en tu computadora (ya lo instalaste), pero ejecutarla no causa problemas.

### Usar VS Code en lugar de Jupyter Lab

Abre la carpeta del repositorio en VS Code, instala las extensiones **Python** y **Jupyter**, abre el notebook y elige el kernel `llm` en la esquina superior derecha (**Select Kernel → Python Environments → llm**).

## Cómo trabajar las prácticas

1. Ejecuta las celdas **en orden**: los ejercicios son acumulativos y algunas pruebas usan el código de ejercicios anteriores.
2. Completa cada sección `TODO` sustituyendo los `None` por tu código.
3. Ejecuta la celda de prueba que sigue. Verás `✅` si obtuviste el punto o `❌` con una pista de qué falló.
4. Si modificas una función, vuelve a ejecutar su celda **y** la celda de prueba.
5. Al final del notebook, la celda **Calificación** muestra tu total.

Esta práctica lee el poema *En paz*, incluido en el repositorio como `en_paz.txt`. No hace falta descargarlo.

## Mantener tu trabajo al día

Cuando se publique una nueva práctica, trae los cambios sin perder tu trabajo. Antes de hacerlo, **guarda tus notebooks con otro nombre** (por ejemplo `02_01_preprocesamiento_texto_llm_MiNombre.ipynb`) o en otra carpeta, y luego ejecuta:

```bash
git pull
```

## Problemas frecuentes

**`ModuleNotFoundError: No module named 'torch'` (o `tiktoken`) en Jupyter**
El notebook está usando otro entorno. Cierra Jupyter, ejecuta `conda activate llm` y vuelve a abrir `jupyter lab` desde esa terminal. En VS Code, revisa que el kernel seleccionado sea `llm`.

**`conda: command not found` o `conda` no se reconoce**
En Windows usa **Anaconda Prompt**. En macOS/Linux ejecuta `conda init` y cierra y vuelve a abrir la terminal.

**`FileNotFoundError` al leer `en_paz.txt`**
El poema está en la carpeta del repositorio, junto al notebook. Ábrelo desde esa carpeta y no muevas `en_paz.txt`.

**`NameError: name '...' is not defined`**
Ejecutaste una celda antes que otra de la que depende. Usa **Run → Run All Above Selected Cell** (o reinicia el kernel y ejecuta todo en orden).

**Quiero empezar desde cero un entorno roto**

```bash
conda deactivate
conda remove -n llm --all -y
```

y repite los pasos desde el punto 3.

## Créditos

Las prácticas sobre LLMs se basan en el libro *Build a Large Language Model (From Scratch)* de Sebastian Raschka (Manning, 2024).

@juan1rving
