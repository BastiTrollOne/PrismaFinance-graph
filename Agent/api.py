import os
import requests
import re
import time
import traceback
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel

# --- IMPORTS ---
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Conexión con el motor de ingesta
from graph_agent import run_graph_extraction

app = FastAPI(title="PrismaFinance Nuclear Agent")

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "prismafinance123")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
CHAT_MODEL = "qwen2.5:3b"

# ==============================================================================
# PROMPT DE INGENIERÍA ROBUSTA
# ==============================================================================
SYSTEM_PROMPT = """You are the Chief Data Architect for PrismaFinance.
Your goal: Generate Cypher queries that find HIDDEN CONNECTIONS in a hybrid graph.

# PROTOCOLS FOR ROBUST QUERYING:

### 1. Safe Type Handling (CRITICAL)
- Some nodes might have merged IDs. ALWAYS use `toString()` before `toLower()`.
- **Pattern:** `WHERE toLower(toString(n.id)) CONTAINS 'search_term'`

### 2. The "Star Search" Pattern
When a user asks about a Person and multiple topics (Money + Text), search for the Person and EVERYTHING connected to them.
- **Pattern:** `MATCH (p:Persona)-[*1..3]-(target) WHERE toLower(toString(p.id)) CONTAINS 'name' AND ...`

### 3. Money Polymorphism
- Money nodes are :Monto, :Costo, :Presupuesto.
- Filter: `(labels(target) IN [['Monto'], ['Costo'], ['Presupuesto']])`

# FEW-SHOT EXAMPLES:

Input: "Pedro Maza gastos flota problemas operativos"
Query: 
MATCH (p:Persona)-[*1..3]-(target) 
WHERE toLower(toString(p.id)) CONTAINS 'pedro maza' 
  AND (
    toLower(toString(target.id)) CONTAINS 'flota' 
    OR toLower(toString(target.id)) CONTAINS 'problema' 
    OR labels(target) IN [['Monto'], ['Costo'], ['Presupuesto']]
  )
RETURN p.id, labels(target), target.id LIMIT 50

Input: "¿Quién dio los 2 millones?"
Query: 
MATCH (n)-[*1..2]-(m) 
WHERE (labels(m) IN [['Monto'], ['Costo'], ['Presupuesto']]) 
  AND toString(m.id) CONTAINS '2' 
RETURN n.id, m.id

Schema:
{schema}
"""

class QueryRequest(BaseModel):
    query: str

# ==============================================================================
# UTILIDADES
# ==============================================================================
def clean_cypher(text: str) -> str:
    pattern = r"```(?:cypher)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    text = text.replace("Cypher:", "").replace("cypher:", "")
    if "MATCH" in text:
        text = text[text.find("MATCH"):]
    return text.strip()

@app.get("/health")
def health():
    return {"status": "active", "mode": "self_healing_v1"}

# ==============================================================================
# ENDPOINT DE CHAT (CON AUTO-REPARACIÓN)
# ==============================================================================
@app.post("/chat")
async def chat(request: QueryRequest):
    print(f"🔥 [CHAT] Pregunta: {request.query}")
    try:
        # 1. Conexión
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        try: graph.refresh_schema()
        except: pass
        
        # 2. Razonamiento
        llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)
        schema_summary = graph.schema[:2000] if graph.schema else "Schema unavailable"
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(schema=schema_summary)),
            HumanMessage(content=f"Query: {request.query}")
        ]
        
        print("   ⏳ Generando query...")
        response_msg = llm.invoke(messages)
        cypher_query = clean_cypher(response_msg.content)
        print(f"   🔧 Query: {cypher_query}")
        
        if not cypher_query:
            return {"response": "Error: Consulta no generada."}

        # 3. Ejecución con AUTO-CURACIÓN (SELF-HEALING)
        print("   ⚡ Ejecutando en Neo4j...")
        results = []
        try:
            results = graph.query(cypher_query)
        except Exception as db_err:
            error_str = str(db_err)
            # DETECCIÓN DEL ERROR DE LISTAS ("StringArray")
            if "StringArray" in error_str or "expected a string" in error_str.lower():
                print("   🚑 ¡ALERTA DE DATOS SUCIOS! Detectada lista en propiedad ID.")
                print("   🛠️ Iniciando protocolo de reparación automática de DB...")
                
                # Query de cirugía: Convierte todas las listas en strings (toma el primer elemento)
                fix_query = """
                MATCH (n) WHERE apoc.meta.type(n.id) = 'LIST'
                SET n.id = toString(n.id[0])
                RETURN count(n) as fixed_nodes
                """
                try:
                    fix_result = graph.query(fix_query)
                    print(f"   ✅ Reparación exitosa. Nodos arreglados: {fix_result}")
                    
                    # REINTENTO INMEDIATO
                    print("   🔄 Reintentando consulta original...")
                    results = graph.query(cypher_query)
                except Exception as fix_err:
                    return {"response": f"Error crítico intentando reparar la DB: {fix_err}"}
            else:
                # Si es otro error (sintaxis real), fallamos
                return {"response": f"Error de sintaxis Cypher: {db_err}"}

        print(f"   🔎 Resultados: {len(results)}")

        if not results:
            return {"response": "No encontré datos conectados. Intenta simplificar la pregunta."}

        # 4. Síntesis
        print("   🗣️ Sintetizando...")
        synthesis_prompt = f"""
        You are a Forensic Financial Auditor.
        USER QUESTION: "{request.query}"
        EVIDENCE: {str(results)}
        
        MISSION:
        1. Answer the question based ONLY on the EVIDENCE.
        2. Connect Person -> Issue -> Money if visible.
        3. Be authoritative. Use Spanish.
        """
        final_response = llm.invoke(synthesis_prompt)
        return {"response": final_response.content}

    except Exception as e:
        print(f"❌ Error Crítico: {e}")
        return {"response": f"Error interno: {str(e)}"}

# ==============================================================================
# ENDPOINTS DE CARGA
# ==============================================================================
def process_graph_in_background(file_path: str):
    print(f"🔄 [BG] Procesando: {file_path}")
    try: run_graph_extraction(file_path)
    except Exception as e: print(f"❌ [BG] Error: {e}")

@app.api_route("/process", methods=["POST", "PUT"])
@app.api_route("/upload", methods=["POST", "PUT"])
async def proxy_upload(request: Request, background_tasks: BackgroundTasks):
    print(f"📥 [UPLOAD] Recibiendo...")
    content_type = request.headers.get("content-type", "").lower()
    filename = request.headers.get("x-filename", "upload.bin")
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
    if not file_content: raise HTTPException(400, "Vacío")

    backup_dir = "/app/backups"
    os.makedirs(backup_dir, exist_ok=True)
    clean_name = re.sub(r'[^\w\-_\.]', '_', filename)
    local_path = f"{backup_dir}/{int(time.time())}_{clean_name}"
    with open(local_path, "wb") as f: f.write(file_content)

    background_tasks.add_task(process_graph_in_background, local_path)
    return {"status": "processing", "file": filename}