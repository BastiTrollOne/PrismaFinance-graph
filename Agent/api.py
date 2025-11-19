import os
import requests
from fastapi import FastAPI, Request, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel
from typing import Optional

# Librerías para el Chat con Grafo (Versiones Modernas)
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import OllamaLLM as Ollama  # Renombramos para no romper tu código abajo
from langchain_core.prompts import PromptTemplate

# Importamos tu lógica de extracción de grafos
from graph_agent import run_graph_extraction

app = FastAPI(title="PrismaFinance Unified API")

# --- CONFIGURACIÓN ---
INGESTION_SERVICE = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
INTERNAL_UPLOAD_URL = f"{INGESTION_SERVICE}/upload"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "prismafinance123")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
# Usamos el modelo rápido para el chat interactivo
CHAT_MODEL = "qwen2.5:3b"


# --- PROMPT MAESTRO BLINDADO (FLEXIBLE + ANTI-SQL) ---
CYPHER_GENERATION_PROMPT = """
Task: Generate a Neo4j Cypher query.
CRITICAL: Do NOT use SQL syntax (no SELECT, no JOIN, no FROM). Use only MATCH, RETURN, WHERE.

Instructions:
1. **Flexible Search (CRITICAL):** Do NOT assume specific node labels like :Persona or :Proyecto unless you are 100% sure. 
   - Use generic `MATCH (n)` instead of specific `MATCH (n:Label)`.
   - ALWAYS use case-insensitive matching: `toLower(n.id) CONTAINS 'term'`.
2. **Traversal Strategy:** Information is often connected via intermediate nodes.
   - Use variable length paths `-[*1..4]-` to find connections across the graph.
   - Example: `MATCH (n)-[*1..4]-(target) WHERE toLower(n.id) CONTAINS 'juan' RETURN n, target`
3. **Return Format:** Return the nodes and relationships found. Do not try to aggregate with SQL functions.

Examples of Valid Queries:
# Q: "¿Presupuesto de Juan?" (Note: No assumptions about labels)
MATCH (n)-[*1..4]-(m)
WHERE toLower(n.id) CONTAINS 'juan' AND (toLower(m.id) CONTAINS 'usd' OR toLower(m.id) CONTAINS '$')
RETURN n, m

# Q: "¿Quién es el responsable?"
MATCH (n)-[*1..2]-(cargo)
WHERE toLower(cargo.id) CONTAINS 'gerente' OR toLower(cargo.id) CONTAINS 'responsable'
RETURN n, cargo

Schema:
{schema}

The question is:
{question}
"""

class QueryRequest(BaseModel):
    query: str

def process_graph_in_background(file_path: str):
    """Tarea en segundo plano: Extraer grafo sin bloquear"""
    print(f"🔄 [Background] Iniciando extracción de grafo para: {file_path}")
    try:
        run_graph_extraction(file_path)
        print(f"✅ [Background] Grafo actualizado con éxito para: {file_path}")
    except Exception as e:
        print(f"❌ [Background] Error procesando grafo: {e}")

@app.get("/health")
def health():
    return {"status": "active", "mode": "unified_graph_rag"}

# --- ENDPOINT DE CHAT (Con Búsqueda Flexible) ---
@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Recibe una pregunta, genera Cypher flexible y consulta Neo4j.
    """
    print(f"💬 Pregunta recibida: {request.query}")
    try:
        # 1. Conectar al Grafo
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        graph.refresh_schema()
        
        # 2. Conectar al LLM
        llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)
        
        # 3. Configurar el Prompt Flexible
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_PROMPT
        )

        # 4. Crear Cadena con el prompt personalizado
        chain = GraphCypherQAChain.from_llm(
            llm=llm, 
            graph=graph, 
            verbose=True, 
            allow_dangerous_requests=True,
            cypher_prompt=cypher_prompt 
        )
        
        # 5. Ejecutar
        result = chain.invoke({"query": request.query})
        print(f"💡 Respuesta generada: {result['result']}")
        return {"response": result['result']}
        
    except Exception as e:
        print(f"❌ Error en chat: {e}")
        return {"response": f"Lo siento, no pude consultar el grafo. Error técnico: {str(e)}"}

# --- ENDPOINT DE SUBIDA (ROBUSTO) ---
@app.api_route("/process", methods=["POST", "PUT"])
@app.api_route("/upload", methods=["POST", "PUT"])
@app.api_route("/process/process", methods=["POST", "PUT"])
async def proxy_upload(request: Request, background_tasks: BackgroundTasks):
    print(f"📥 Recibiendo solicitud de archivo en: {request.url.path}")

    # 1. Extraer el archivo (Lógica robusta)
    content_type = request.headers.get("content-type", "").lower()
    filename = "doc_sin_nombre.bin"
    file_content = b""
    file_object = None

    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
            # Buscar en campos comunes
            for key in ["file", "files", "doc", "document"]:
                if key in form:
                    candidate = form[key]
                    if isinstance(candidate, (UploadFile, StarletteUploadFile)):
                        file_object = candidate
                        break
            # Búsqueda genérica
            if not file_object:
                for _, val in form.multi_items():
                    if isinstance(val, (UploadFile, StarletteUploadFile)):
                        file_object = val
                        break
            
            if file_object:
                filename = file_object.filename or filename
                file_content = await file_object.read()
                content_type = file_object.content_type or content_type
        except Exception as e:
            print(f"⚠️ Error leyendo form-data: {e}")
    
    if not file_content:
        # Intento desesperado: leer body crudo
        try:
            file_content = await request.body()
            if request.headers.get("x-filename"):
                filename = request.headers.get("x-filename")
        except:
            pass

    if not file_content:
        raise HTTPException(400, "No se encontró archivo en la petición.")

    print(f"📄 Procesando: {filename}")

    # 2. Guardar localmente para el Agente
    temp_path = f"/app/{filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        print(f"❌ Error guardando temporal: {e}")

    # 3. Reenviar a Ingesta
    try:
        files_to_send = {'file': (filename, file_content, content_type)}
        params = dict(request.query_params)
        
        print(f"📤 Reenviando a Ingesta: {INTERNAL_UPLOAD_URL}")
        response = requests.post(INTERNAL_UPLOAD_URL, files=files_to_send, params=params)
        
        if response.status_code != 200:
            return JSONResponse(status_code=response.status_code, content=response.json())
            
        result_json = response.json()
        
        # 4. Activar Grafo en Background
        background_tasks.add_task(process_graph_in_background, temp_path)
        
        return result_json

    except Exception as e:
        print(f"❌ Error en proxy: {e}")
        raise HTTPException(500, detail=str(e))