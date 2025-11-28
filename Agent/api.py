import os
import time
import traceback
import uuid
import re          # <--- AGREGADO
import requests    # <--- AGREGADO
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel
from typing import List, Optional

# --- IMPORTS ---
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph_agent import run_graph_extraction

app = FastAPI(title="PrismaFinance Nuclear Agent (OpenAI Compatible)")

# --- CONFIGURACIÓN ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "prismafinance123")

# Conexión al LLM Remoto (El cerebro real)
OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://192.168.50.1:8900/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-needed")
# Modelo que usaremos en el remoto
REMOTE_MODEL_ID = os.getenv("LLM_MODEL_ID", "unsloth/qwen3-4b-instruct-2507")

INGESTION_SERVICE = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
INTERNAL_UPLOAD_URL = f"{INGESTION_SERVICE}/upload"

# --- PROMPTS ---
EXTRACTION_PROMPT = """
You are an Expert Search Term Extractor for a Financial Knowledge Graph.
Your goal is to identify the single most important "anchor" term from the user's query to query a Neo4j database.

DATABASE SCHEMA:
- Personas (e.g., 'Pedro Maza', 'Ana Rojas')
- Organizaciones (e.g., 'Metso', 'Candelaria')
- Proyectos (e.g., 'Mantenimiento Planta', 'Candelaria 2030')
- Conceptos (e.g., 'Bono', 'Presupuesto', 'Auditoría')

RULES:
1. Identify the MAIN SUBJECT. It can be a Person, Organization, Project, or Concept.
2. CLEAN THE TERM: Remove stop words ("el", "la", "de", "sobre") and action verbs ("reporto", "gastó", "dijo").
3. DO NOT output labels like "Person:" or "Project:". Just the raw term.
4. IF multiple entities exist, prioritize the most specific proper name (Person > Project > Organization).
5. IF the query is general (e.g., "resumen del mes"), output "GENERAL".

EXAMPLES:
Input: "Gastos de viaje de Ana Rojas"
Output: Ana Rojas

Input: "¿Cuánto presupuesto tiene el Proyecto Candelaria 2030?"
Output: Candelaria 2030

Input: "Pagos realizados a Metso Outotec"
Output: Metso Outotec

Input: "Problemas con la flota Komatsu"
Output: Komatsu

Input: "Dime todo sobre los bonos"
Output: Bono
"""

# --- LÓGICA CENTRAL DEL AGENTE (Separada para reutilizar) ---
async def run_agent_logic(query: str) -> str:
    print(f"🔥 [AGENTE] Procesando: {query}")
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        llm = ChatOpenAI(
            model=REMOTE_MODEL_ID,
            openai_api_base=OPENAI_BASE,
            openai_api_key=OPENAI_KEY,
            temperature=0
        )
        
        # 1. Extracción
        print("   🧠 [1/3] Extrayendo entidad...")
        extraction_messages = [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=f"Input: {query}")
        ]
        entity_name = llm.invoke(extraction_messages).content.strip().replace('"', '').replace("'", "")
        # Limpieza básica
        if len(entity_name) > 30: entity_name = entity_name.split(" sobre")[0]
        print(f"   🎯 Entidad: '{entity_name}'")

        # 2. Búsqueda en Grafo
        cypher_query = f"""
        MATCH (p)-[*1..2]-(related)
        WHERE toLower(toString(p.id)) CONTAINS toLower('{entity_name}')
        RETURN p, related LIMIT 100
        """
        print("   ⚡ [2/3] Consultando Neo4j...")
        results = graph.query(cypher_query)
        
        if not results:
            return f"No encontré registros para '{entity_name}' en la base de datos local."

        # 3. Síntesis
        print("   🗣️ [3/3] Sintetizando respuesta...")
        synthesis_prompt = f"""
        You are a Data Extractor.
        USER QUESTION: "{query}"
        DATABASE RECORDS: {str(results)}
        
        TASK: Answer the question based ONLY on the RECORDS.
        If you find costs/payments, specify amounts and companies.
        Answer in Spanish.
        """
        final_response = llm.invoke(synthesis_prompt).content
        return final_response

    except Exception as e:
        print(f"❌ Error Agente: {e}")
        traceback.print_exc()
        return f"Error interno del sistema: {str(e)}"

# --- ENDPOINTS COMPATIBLES CON OPENAI (Para OWUI nativo) ---

@app.get("/v1/models")
def list_models():
    # Esto hace que 'PrismaFinance-Agent' aparezca en la lista de OWUI
    return {
        "object": "list",
        "data": [{
            "id": "PrismaFinance-Agent",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "prisma-local"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. Recibir formato OpenAI
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "Sin mensajes")
    
    # 2. Extraer la última pregunta del usuario
    user_query = messages[-1].get("content", "")
    
    # 3. Ejecutar lógica del Agente
    response_text = await run_agent_logic(user_query)
    
    # 4. Devolver formato OpenAI
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "PrismaFinance-Agent"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.get("/health")
def health():
    return {"status": "ready", "mode": "openai-compatible"}

# --- ENDPOINTS DE CARGA (Sin cambios, necesarios para Ingesta) ---
@app.api_route("/process", methods=["POST", "PUT"])
@app.api_route("/upload", methods=["POST", "PUT"])
async def proxy_upload(request: Request, background_tasks: BackgroundTasks):
    print(f"📥 [UPLOAD] Solicitud de OWUI...")
    content_type = request.headers.get("content-type", "").lower()
    filename = request.headers.get("x-filename", "doc.bin")
    file_content = b""
    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
            for key, val in form.multi_items():
                if isinstance(val, StarletteUploadFile):
                    filename = val.filename or filename
                    file_content = await val.read()
                    break
        except: pass
    if not file_content:
        try: file_content = await request.body()
        except: pass
    if not file_content: raise HTTPException(400, "Sin archivo")

    backup_dir = "/app/backups"
    os.makedirs(backup_dir, exist_ok=True)
    base_name = os.path.basename(filename)
    name_part, ext_part = os.path.splitext(base_name)
    if not ext_part: ext_part = ".bin"
    clean_name = re.sub(r'[^\w\-_\.]', '_', name_part)[:50]
    final_filename = f"{clean_name}{ext_part}"
    local_path = f"{backup_dir}/{int(time.time())}_{final_filename}"
    
    with open(local_path, "wb") as f: f.write(file_content)

    try:
        files = {'file': (filename, file_content, content_type)}
        res = requests.post(
            INTERNAL_UPLOAD_URL, 
            files=files, 
            params={"chunk_size": 1000, "chunk_overlap": 300}
        )
        if res.status_code == 200:
            background_tasks.add_task(process_graph_in_background, local_path)
            return res.json()
        else:
            return {"page_content": "", "metadata": {"error": "ingestion_failed"}}
    except Exception as e:
        return {"page_content": "", "metadata": {"error": str(e)}}

def process_graph_in_background(file_path: str):
    try: run_graph_extraction(file_path)
    except: pass