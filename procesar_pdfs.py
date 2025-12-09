import os
import pickle
import numpy as np
import shutil
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import re
import nltk
from nltk.corpus import stopwords 

# CONFIGURACIÓN
DOCS_DIR = "docs"
INDEX_FILE = "indice_tfidf.pkl"
MAX_CHARS_CHUNK = 500 

# Asegurar que los recursos de NLTK estén disponibles
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Descargando recursos necesarios de NLTK (punkt, stopwords)...")
    nltk.download('punkt')
    nltk.download('stopwords')
    
# Obtener la lista de stopwords para español (Necesario para TfidfVectorizer)
STOPWORDS_ESPANOL = stopwords.words('spanish')

# ============================
# Extraer texto de PDF y metadatos
# ============================
def extraer_texto_pdf(path):
    try:
        reader = PdfReader(path)
        texto = ""
        # Intentar obtener el título del documento
        titulo = reader.metadata.get('/Title', os.path.basename(path)) 
        
        # Extracción de texto y mapeo de página
        page_chunks = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Almacenamos el texto junto con el número de página (i+1)
                page_chunks.append({
                    "texto": page_text + "\n", 
                    "page_num": i + 1
                })
        
        return {"chunks_por_pagina": page_chunks, "titulo": titulo}
    except Exception as e:
        print(f"Error leyendo {path}: {e}")
        return {"chunks_por_pagina": [], "titulo": os.path.basename(path)}

# ============================
# Cargar todos los documentos
# ============================
def cargar_documentos():
    docs = []
    print(f"\nBuscando PDFs en: {DOCS_DIR}\n")

    for filename in os.listdir(DOCS_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DOCS_DIR, filename)
            print(f"Procesando: {filename}")
            
            data = extraer_texto_pdf(path)
            
            if data["chunks_por_pagina"]:
                docs.append({
                    "chunks_por_pagina": data["chunks_por_pagina"], 
                    "fuente": filename, 
                    "titulo": data["titulo"]
                })

    return docs


# ============================
# Chunking inteligente (basado en oraciones)
# ============================
def hacer_chunks_inteligente(texto_con_pagina, max_chars=MAX_CHARS_CHUNK):
    partes = []
    texto = texto_con_pagina["texto"]
    page_num = texto_con_pagina["page_num"]
    
    # 1. Segmentar el texto en oraciones, especificando español
    oraciones = nltk.sent_tokenize(texto, language='spanish')
    actual = ""

    for oracion in oraciones:
        if len(actual) + len(oracion) + 1 > max_chars:
            if actual:
                # Guardamos el chunk con el número de página de origen
                partes.append({"texto": actual.strip(), "page": page_num})
            
            actual = oracion
        else:
            actual += " " + oracion

    if actual:
        partes.append({"texto": actual.strip(), "page": page_num})

    return partes


# ============================
# Función principal de RE-INDEXACIÓN
# ============================
def reindexar_todo():
    """Carga todos los documentos, procesa, genera TF-IDF y Embeddings, y guarda el índice."""
    
    documentos = cargar_documentos()
    
    print("\nExpandiendo documentos en chunks inteligentes...")
    chunks = []
    metadatos = []

    for doc in documentos:
        for page_data in doc["chunks_por_pagina"]:
            partes = hacer_chunks_inteligente(page_data) 
            for ch in partes:
                chunks.append(ch["texto"])
                # Guardamos metadatos clave: fuente, título, y NÚMERO DE PÁGINA
                metadatos.append({
                    "fuente": doc["fuente"],
                    "titulo": doc["titulo"], 
                    "pagina": ch["page"] # Metadato extra para Objetivo 4
                })

    print(f"Total de chunks: {len(chunks)}")
    if not chunks:
        print("Advertencia: No hay chunks para indexar. Creando índice vacío.")
        indice_data = {
            "documentos": [], "tfidf": np.array([]), "vectorizer": TfidfVectorizer(),
            "embeddings": np.array([])
        }
    else:
        # 1. TF-IDF (Corregido: Usando lista STOPWORDS_ESPANOL de NLTK)
        print("\nGenerando TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=20000, stop_words=STOPWORDS_ESPANOL) 
        tfidf = vectorizer.fit_transform(chunks).toarray().astype("float32")

        # 2. EMBEDDINGS (Uso de GPU forzado)
        print("\nCargando modelo de embeddings (forzando uso de GPU 'cuda' si está disponible)...")
        try:
            # Objetivo 5: Aprovechamiento HW ATY (GPU)
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
            print("Modelo cargado en GPU.")
        except Exception:
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
            print("Modelo cargado en CPU.")


        print("Generando embeddings densos...")
        embeddings = model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
        ).astype("float32")

        # 3. Datos del índice
        indice_data = {
            "documentos": [
                {"texto": ch, **metadatos[i]} 
                for i, ch in enumerate(chunks)
            ],
            "tfidf": tfidf,
            "vectorizer": vectorizer,
            "embeddings": embeddings,
        }

    # 4. Guardar
    print("\nGuardando índice...")
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(indice_data, f)

    print("✔ Índice generado correctamente.")
    return len(chunks)


# ============================
# Funciones de Gestión Administrativa (Objetivo 2)
# ============================

def agregar_pdf(temp_path: str, filename: str):
    """Copia un nuevo PDF al directorio docs y re-indexa."""
    destino = os.path.join(DOCS_DIR, filename)
    
    # Manejo de Reemplazo
    if os.path.exists(destino):
        os.remove(destino) 
        print(f"Documento existente reemplazado: {filename}")
        
    shutil.copyfile(temp_path, destino)
    os.remove(temp_path) 
    
    print(f"Archivo guardado en {destino}")
    
    chunk_count = reindexar_todo()
    return chunk_count

def eliminar_pdf(filename: str):
    """Elimina un PDF del directorio docs y re-indexa."""
    path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        print(f"Documento eliminado: {filename}")
        chunk_count = reindexar_todo()
        return chunk_count
    return 0

def listar_documentos():
    """Devuelve un listado de documentos con sus metadatos básicos."""
    documentos = []
    for filename in os.listdir(DOCS_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DOCS_DIR, filename)
            stat = os.stat(path)
            documentos.append({
                "nombre": filename,
                "tamaño_kb": round(stat.st_size / 1024, 2),
                "fecha_modificacion": stat.st_mtime, 
            })
    return documentos

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    reindexar_todo()