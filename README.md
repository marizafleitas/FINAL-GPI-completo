# Chatbot Normativo FCyT -- Motor Híbrido Avanzado 2025

Este proyecto implementa un **Motor de Recuperación de Información
Híbrida** para la Facultad de Ciencias y Tecnologías (FCyT -- UNCA), que
permite realizar consultas avanzadas sobre reglamentos y documentos
institucionales a partir de archivos PDF.

El objetivo de esta versión es demostrar la implementación de **mejoras
técnicas avanzadas**, como la combinación de búsqueda por palabras clave
y búsqueda por significado (semántica), junto con el Re-ranking, para
obtener resultados de alta precisión.

------------------------------------------------------------------------

## 🧭 ¿Qué hace este sistema?

El proyecto permite consultar documentos normativos utilizando preguntas
en lenguaje natural a través de un **Sistema Híbrido de dos fases**:

1.  **Motor Híbrido (Búsqueda Inicial):**\
    Combina la **matriz TF-IDF** (búsqueda por palabras clave) con los
    **Embeddings Densos** (búsqueda por significado) para seleccionar
    una amplia lista de candidatos relevantes.

2.  **Re-ranking Semántico (Filtrado Final):**\
    Utiliza un modelo de **Sentence Transformers**
    (`paraphrase-multilingual-MiniLM-L12-v2`) para reordenar los
    candidatos basándose en la **similitud semántica pura**,
    garantizando que los fragmentos más precisos y contextualmente
    relevantes lleguen al Top-5 final.

Este enfoque garantiza que el sistema:

-   Responde con alta **precisión semántica**, incluso si el usuario usa
    sinónimos.
-   Es **robusto** ante la ambigüedad, utilizando lo mejor de la
    búsqueda literal y la conceptual.
-   Funciona **completamente offline** una vez instalado el modelo.

------------------------------------------------------------------------

## 🧩 Requisitos

✔ Python 3.11 (recomendado)\
✔ Conexión a internet (solo necesaria la primera vez para descargar el
modelo de embeddings)

------------------------------------------------------------------------

## 🔧 Instalación

### 1. Clonar el repositorio

``` bash
git clone https://github.com/marizafleitas/FINAL-GPI-completo.git
cd fcyt-chatbot-normativo
```

### 2. Crear y activar el entorno virtual

**Windows (PowerShell):**

``` bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

**Importante:** Esta versión instala librerías para la búsqueda
semántica (`sentence-transformers`):

``` bash
pip install -r requirements.txt
```

Esto instala:

-   fastapi, uvicorn, pydantic\
-   pypdf\
-   numpy, scikit-learn\
-   sentence-transformers (para embeddings)

------------------------------------------------------------------------

## 📁 Estructura del proyecto

    fcyt-chatbot-normativo/
    ├─ app.py                # Servidor FastAPI y Rutas
    ├─ chatbot.py            # Script de consulta en consola
    ├─ procesar_pdfs.py      # Script de pre-procesamiento e indexación
    ├─ requirements.txt
    ├─ docs/                 # PDFs normativos de entrada
    └─ .gitignore

------------------------------------------------------------------------

## ⚙ Procesar los PDFs (generar el índice híbrido)

Antes de cualquier consulta, se debe generar el índice que contendrá la
información para las dos técnicas de búsqueda:

``` bash
python procesar_pdfs.py
```

Esto produce un archivo `indice_tfidf.pkl` que contiene:

-   Fragmentos de texto\
-   Vectorizador TF-IDF\
-   Matriz de embeddings semánticos para todos los fragmentos

Cada vez que se agreguen o cambien PDFs en `docs/`, se debe ejecutar
nuevamente este comando.

------------------------------------------------------------------------

## 💬 Uso del chatbot en modo consola

``` bash
python chatbot.py
```

El sistema devolverá los fragmentos más relevantes y el documento
correspondiente, priorizando la precisión semántica.

------------------------------------------------------------------------

## 🌐 Servidor web con FastAPI

Levantar el servidor:

``` bash
uvicorn app:app --reload --port 8000
```

Abrir en el navegador:

    http://127.0.0.1:8000/

La interfaz mostrará:

-   Score híbrido\
-   Score semántico final\
-   Fragmento recuperado\
-   Origen: página y documento

Para detener el servidor:

    CTRL + C

------------------------------------------------------------------------

## 📄 Licencia y uso académico

Este proyecto está diseñado para fines educativos dentro de la FCyT --
UNCA.
