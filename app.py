import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer # Importación necesaria
import operator

# Se asume que esta ruta de administración existe y funciona correctamente
from admin.routes import router as admin_router 

# ============================
# Cargar índice y MODELO GLOBALMENTE (Objetivo 5: Rendimiento)
# ============================
print("Cargando índice TF-IDF...")
with open("indice_tfidf.pkl", "rb") as f:
    data = pickle.load(f)

documentos = data["documentos"]
vectorizer = data["vectorizer"]
tfidf_matrix = data["tfidf"]
embeddings = data["embeddings"] 

# ¡CARGA CRÍTICA! Cargar el modelo de SBERT una sola vez y en GPU
print("Cargando modelo SentenceTransformer en memoria (forzando uso de GPU)...")
try:
    # Usamos 'cuda' para aprovechar la GPU del Proyecto ATY
    SBERT_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    print("Modelo SBERT cargado en GPU (CUDA).")
except Exception:
    SBERT_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    print("Modelo SBERT cargado en CPU.")


# Modelo del usuario
class Question(BaseModel):
    question: str


# ============================
# Búsqueda híbrida TF-IDF + Embeddings (con re-ranking opcional)
# ============================
def buscar_respuesta(pregunta: str, k_base=20, k_final=5, alpha=0.3):
    # 1. Vector TF-IDF
    q_tfidf = vectorizer.transform([pregunta]).toarray().astype("float32")

    # Similaridad TF-IDF
    sims_tfidf = cosine_similarity(q_tfidf, tfidf_matrix)[0]

    # 2. Embedding denso (Usando el modelo cargado globalmente y en GPU)
    q_emb = SBERT_MODEL.encode([pregunta], convert_to_numpy=True, normalize_embeddings=True)
    sims_emb = cosine_similarity(q_emb, embeddings)[0]

    # 3. Combinación híbrida
    # Normalización Min-Max (mejor para este tipo de combinación simple)
    sims_tfidf = (sims_tfidf - sims_tfidf.min()) / (sims_tfidf.max() - sims_tfidf.min())
    sims_emb = (sims_emb - sims_emb.min()) / (sims_emb.max() - sims_emb.min())
    
    score_hibrido = alpha * sims_tfidf + (1 - alpha) * sims_emb


    # 4. Fase de Re-ranking (Top-k base)
    idxs_base = np.argsort(score_hibrido)[::-1][:k_base]
    
    # 5. Re-ranking: Se usa el score de embeddings (mayor peso semántico).
    scores_candidatos = {idx: sims_emb[idx] for idx in idxs_base}
    
    # Ordenar los candidatos por el score de embeddings
    sorted_candidates = sorted(scores_candidatos.items(), key=operator.itemgetter(1), reverse=True)
    
    # Seleccionar los k_final mejores
    final_idxs = [idx for idx, score in sorted_candidates][:k_final]

    
    # 6. Formateo de Resultados
    resultados = [
        {
            "texto": documentos[i]["texto"],
            "fuente": documentos[i]["fuente"],
            "titulo": documentos[i]["titulo"], # Identificación de artículo
            # Se asume que 'pagina' está en el índice. Si no lo está, la interfaz fallará.
            "pagina": documentos[i].get("pagina", "N/A"), 
            "score_hibrido": float(score_hibrido[i]),
            "score_re_ranking": float(sims_emb[i]), # Usamos el score del re-ranking
        }
        for i in final_idxs
    ]

    return resultados


# ============================
# FastAPI APP
# ============================
app = FastAPI(title="Chatbot FCyT - Motor híbrido avanzado")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(admin_router, prefix="/admin")


# ============================
# Interfaz de usuario mejorada (Objetivo 4)
# ============================
@app.get("/", response_class=HTMLResponse)
def home():
    # Hemos mejorado la interfaz para mostrar más información (score, título)
    return """
    <html>
    <head>
        <title>Chatbot Normativo FCyT</title>
        <meta charset="utf-8">
        <style>
            .resultado {
                margin: 15px 0; 
                padding: 15px; 
                border-left: 5px solid #007bff; 
                background-color: #f8f9fa;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .meta {
                font-size: 0.8em; 
                color: #6c757d;
                margin-top: 5px;
            }
        </style>
    </head>
    <body style="font-family: sans-serif; max-width: 900px; margin: auto; padding: 20px;">
        <h1>Chatbot Normativo FCyT 🧪</h1>
        <p>Sistema **Híbrido Avanzado**: TF-IDF + Embeddings + Re-ranking.</p>
        <p>Hardware ATY utilizado para indexación y búsqueda acelerada por GPU.</p>

        <textarea id="q" style="width:100%; height:80px; padding: 10px; border: 1px solid #ccc;"></textarea><br>
        <button onclick="ask()" style="padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer;">Consultar</button>

        <div id="resp"></div>

        <script>
            async function ask() {
                let q = document.getElementById("q").value;
                let div = document.getElementById("resp");
                div.innerHTML = "<p>Buscando, espere un momento...</p>";

                let r = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({question: q})
                });

                let data = await r.json();
                div.innerHTML = "";

                data.resultados.forEach((res, i) => {
                    // INICIO DE LAS MODIFICACIONES SOLICITADAS
                    div.innerHTML += `
                        <div class='resultado'>
                            <h4>[${i+1}] (Pág. ${res.pagina}) ${res.titulo}</h4> 
                            <p>${res.texto}</p>
                            <div class="meta">
                                <b>Fuente:</b> ${res.fuente} | 
                                <b>Score Final (Semántica Pura):</b> ${res.score_re_ranking.toFixed(3)} |
                                <b>Score Base (Híbrida TF-IDF+Emb):</b> ${res.score_hibrido.toFixed(3)}
                            </div>
                        </div>
                    `;
                    // FIN DE LAS MODIFICACIONES SOLICITADAS
                });
            }
        </script>

        <p><a href="/admin">Ir al panel administrativo</a></p>
    </body>
    </html>
    """


@app.post("/ask")
def ask(q: Question):
    # Se usan los parámetros k_base=20 y k_final=5 para el re-ranking
    res = buscar_respuesta(q.question, k_base=20, k_final=5, alpha=0.3)
    return {"resultados": res}