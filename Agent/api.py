import os
import requests
import re
import time
import traceback
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel

# --- IMPORTS ---
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from graph_agent import run_graph_extraction

app = FastAPI(title="PrismaFinance Nuclear Agent")

# CONFIGURACIÓN
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "prismafinance123")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
CHAT_MODEL = "qwen2.5:3b"
INGESTION_SERVICE = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
INTERNAL_UPLOAD_URL = f"{INGESTION_SERVICE}/upload"

# --- 1. PROMPT DE EXTRACCIÓN (EL CEREBRO DE BÚSQUEDA) ---
# (Este NO lo tocamos, ya funciona bien)
EXTRACTION_PROMPT = """You are a Named Entity Extractor.
Task: Extract the main Person or Organization name from the user query.
Rules:
1. Output ONLY the name. No explanations. No punctuation.
2. If multiple entities, pick the most specific Person.
3. Remove words like "gastos", "roles", "reporto", "problemas".

Example:
Input: "Que problemas reporto Pedro Maza sobre la flota"
Output: Pedro Maza

Input: "Gastos de Ana Rojas"
Output: Ana Rojas
"""

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "active", "mode": "final_audit_v9"}

@app.post("/chat")
async def chat(request: QueryRequest):
    print(f"🔥 [CHAT] Pregunta: {request.query}")
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)
        
        # --- FASE 1: EXTRACCIÓN DE ENTIDAD ---
        print("   🧠 Extrayendo entidad con LLM...")
        extraction_messages = [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=f"Input: {request.query}")
        ]
        entity_name_raw = llm.invoke(extraction_messages).content
        
        # Limpieza final
        entity_name = entity_name_raw.replace("\n", "").replace('"', '').replace("'", "").strip()
        if len(entity_name) > 30 or " " not in entity_name: 
            # Fallback por si el modelo alucina frases largas
            print(f"   ⚠️ Extracción dudosa ('{entity_name}'), aplicando filtro manual...")
            entity_name = entity_name.split(" sobre")[0].split(" report")[0] 
            
        print(f"   🎯 Entidad Identificada: '{entity_name}'")
        
        # --- FASE 2: INYECCIÓN SEGURA EN CYPHER ---
        # Buscamos el nodo y TODO lo conectado a 1 salto (proyectos, costos, documentos)
        cypher_query = f"""
        MATCH (p)-[*1..2]-(related)
        WHERE toLower(toString(p.id)) CONTAINS toLower('{entity_name}')
        RETURN p, related LIMIT 100
        """
        
        print(f"   ⚡ Ejecutando en Neo4j...")
        results = graph.query(cypher_query)
        print(f"   🔎 Resultados: {len(results)}")

        if not results:
            return {"response": f"No encontré información exacta para '{entity_name}' en el grafo financiero."}

        # --- FASE 3: SÍNTESIS (AQUÍ ESTÁ EL CAMBIO) ---
        print("   🗣️ Sintetizando...")
        
        # Este es el nuevo prompt agresivo para que lea los Excel
        synthesis_prompt = f"""
        You are a Data Extractor. Do not interpret. Just extract facts.
        USER QUESTION: "{request.query}"
        DATABASE RECORDS: {str(results)}
        
        TASK:
        1. Look for the person '{entity_name}' in the RECORDS.
        2. Find any associated MONEY AMOUNTS (look for labels 'Monto', 'Costo', 'Presupuesto' or numbers like 25000000, 3500000).
        3. Find associated COMPANIES (e.g. AES Andes, Caterpillar) or PROJECTS.
        4. If you find a cost/payment, say: "Según los registros, [Persona] gestionó un pago de [Monto] a [Empresa] para [Concepto]."
        5. If the data comes from a Document (text), summarize the role/problem.
        
        IMPORTANT: If you see the data in the RECORDS, report it immediately. Do not ask for more details.
        Answer in Spanish.
        """
        
        final_response = llm.invoke(synthesis_prompt)
        return {"response": final_response.content}

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"response": f"Error técnico: {str(e)}"}

# --- RUTAS DE UPLOAD (Sin cambios) ---
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