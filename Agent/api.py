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

# Importación del motor
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

# PROMPT MAESTRO
SYSTEM_PROMPT = """You are the Supreme Data Architect for PrismaFinance.
Your mission is to generate Cypher queries that retrieve data regardless of structural ambiguity.

# GLOBAL SEARCH PROTOCOLS (THE "GOD MODE" RULES):

### 1. Entity Agnosticism (NEVER ASSUME TYPES)
User questions are ambiguous.
- **GOD MODE:** `MATCH (n:Persona|Organizacion|Proyecto|Concepto) WHERE toLower(toString(n.id)) CONTAINS 'term'`

### 2. The "Double Anchor" Strategy
- Find ALL paths connecting Entity A to Entity B.
- **Pattern:** `MATCH (a:Persona|Organizacion)-[*1..4]-(b:Persona|Organizacion|Concepto|Monto)`

### 3. Financial Dragnet
- **Filter:** `(labels(m) IN [['Monto'], ['Costo'], ['Presupuesto']])`
- **Value:** `m.id CONTAINS '2000'`

### 4. Syntax Safety
- Always use `toString()` before `toLower()`.
- Always wrap `OR` conditions in parentheses.

# MASTER EXAMPLES:

Input: "Javiera Silva incidente Transportes Tamarugal"
Query: 
MATCH (a)-[*1..4]-(b)
WHERE (toLower(toString(a.id)) CONTAINS 'javiera')
  AND (toLower(toString(b.id)) CONTAINS 'tamarugal')
RETURN a.id, labels(a), b.id, labels(b) LIMIT 50

Input: "Quién representa al banco que dio los 2 millones?"
Query: 
MATCH (p:Persona)-[*1..4]-(o:Organizacion)-[*1..4]-(m)
WHERE toLower(toString(o.id)) CONTAINS 'banco'
  AND (labels(m) IN [['Monto'], ['Costo'], ['Presupuesto']])
  AND (m.id CONTAINS '2000' OR m.id CONTAINS '2')
RETURN p.id, o.id, m.id

Schema:
{schema}
"""

class QueryRequest(BaseModel):
    query: str

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
    return {"status": "active", "mode": "extension_fix_v4"}

@app.post("/chat")
async def chat(request: QueryRequest):
    print(f"🔥 [CHAT] Pregunta: {request.query}")
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        try: graph.refresh_schema()
        except: pass
        
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
        
        if not cypher_query: return {"response": "Error: Consulta vacía."}

        print("   ⚡ Ejecutando en Neo4j...")
        try:
            results = graph.query(cypher_query)
        except Exception as db_err:
            error_str = str(db_err)
            if "StringArray" in error_str or "expected a string" in error_str.lower():
                print("   🚑 Auto-reparando DB...")
                try:
                    graph.query("MATCH (n) WHERE apoc.meta.type(n.id) = 'LIST' SET n.id = toString(n.id[0])")
                    results = graph.query(cypher_query)
                except: return {"response": f"Error crítico DB: {db_err}"}
            else:
                return {"response": f"Error de sintaxis: {db_err}"}

        print(f"   🔎 Resultados: {len(results)}")

        if not results:
            return {"response": "No encontré datos conectados."}

        print("   🗣️ Sintetizando...")
        synthesis_prompt = f"""
        You are an expert Financial Auditor.
        QUERY: "{request.query}"
        EVIDENCE: {str(results)}
        
        INSTRUCTIONS:
        1. Connect the dots based ONLY on the evidence.
        2. If you see a Person linked to an Organization or Incident, explain the connection.
        3. Be precise with names and amounts.
        4. Answer in Spanish.
        """
        final_response = llm.invoke(synthesis_prompt)
        return {"response": final_response.content}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"response": f"Error interno: {str(e)}"}

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

    # === CORRECCIÓN DE NOMBRE LARGO (PRESERVANDO EXTENSIÓN) ===
    backup_dir = "/app/backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    # 1. Separar nombre y extensión
    base_name = os.path.basename(filename)
    name_part, ext_part = os.path.splitext(base_name)
    if not ext_part: ext_part = ".bin" # Fallback si no tiene extensión
    
    # 2. Limpiar y recortar solo el nombre
    clean_name = re.sub(r'[^\w\-_\.]', '_', name_part)
    if len(clean_name) > 50:
        clean_name = clean_name[:50]
    
    # 3. Reconstruir nombre corto CON extensión original
    final_filename = f"{clean_name}{ext_part}"
    local_path = f"{backup_dir}/{int(time.time())}_{final_filename}"
    
    print(f"   💾 Guardando como: {final_filename}")
    with open(local_path, "wb") as f: f.write(file_content)

    # Ingesta Sincrónica para OWUI
    try:
        print(f"   ⚡ Enviando a Ingesta (Sync) para OWUI...")
        files = {'file': (filename, file_content, content_type)}
        res = requests.post(
            INTERNAL_UPLOAD_URL, 
            files=files, 
            params={"chunk_size": 1000, "chunk_overlap": 300}
        )
        
        if res.status_code == 200:
            background_tasks.add_task(process_graph_in_background, local_path)
            print("   ✅ Retornando JSON válido a OWUI.")
            return res.json()
        else:
            return {"page_content": "", "metadata": {"error": "ingestion_failed"}}

    except Exception as e:
        print(f"❌ Error en proxy_upload: {e}")
        return {"page_content": "", "metadata": {"error": str(e)}}

def process_graph_in_background(file_path: str):
    print(f"🔄 [BG] Disparando graph_agent para: {file_path}")
    try: run_graph_extraction(file_path)
    except: pass