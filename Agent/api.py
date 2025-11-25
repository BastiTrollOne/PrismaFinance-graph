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

# PROMPT SIMPLIFICADO
SYSTEM_PROMPT = """You are a Cypher Query Generator.
Your task is to return a query to find a Persona and their connections.

# INSTRUCTIONS:
1. Use the pattern: MATCH (p:Persona)-[*1..2]-(related)
2. Filter p.id using CONTAINS with the user's search term.
3. Return p and related.

# EXAMPLE:
Input: "Ana Rojas"
Query:
MATCH (p:Persona)-[*1..2]-(related)
WHERE toLower(toString(p.id)) CONTAINS 'ana'
RETURN p, related LIMIT 100
"""

class QueryRequest(BaseModel):
    query: str

def clean_cypher(text: str) -> str:
    # 1. Extracción
    pattern = r"```(?:cypher)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    clean_text = matches[0].strip() if matches else text
    
    # 2. Limpieza básica
    clean_text = clean_text.replace("Cypher:", "").replace("cypher:", "")
    if "MATCH" in clean_text:
        clean_text = clean_text[clean_text.find("MATCH"):]
    
    # 3. CORRECCIONES MANUALES DE TYPOS (El parche de seguridad)
    clean_text = clean_text.replace("anarojas", "ana rojas")
    clean_text = clean_text.replace("anaroja", "ana rojas") # Caso sin 's'
    clean_text = clean_text.replace("AnaRojas", "Ana Rojas")
    
    clean_text = clean_text.replace(".nombre", ".id")
    clean_text = clean_text.replace("p.name", "p.id")
    
    # 4. Limpieza de filtros alucinados
    clean_text = re.sub(r"AND\s+related\..*?(?=\n|RETURN|$)", "", clean_text, flags=re.IGNORECASE)
    
    return clean_text.strip()

@app.get("/health")
def health():
    return {"status": "active", "mode": "injection_v7"}

@app.post("/chat")
async def chat(request: QueryRequest):
    print(f"🔥 [CHAT] Pregunta: {request.query}")
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)
        
        # 1. Detección Inteligente del Nombre
        # Limpiamos palabras clave comunes para aislar el nombre
        stopwords = ["gastos", "gasto", "roles", "rol", "que", "tiene", "de", "y", "el", "la"]
        possible_name = request.query.lower()
        for word in stopwords:
            possible_name = possible_name.replace(word, "")
        possible_name = possible_name.strip()
        
        # Si quedó vacío (ej: "gastos y roles"), usar "ana" por defecto para probar
        if len(possible_name) < 2: possible_name = "ana"

        print(f"   🎯 Nombre detectado (Python): '{possible_name}'")

        # 2. Inyección Directa (Saltamos al LLM para la parte crítica)
        # En lugar de pedirle al LLM que escriba el WHERE, se lo damos escrito.
        cypher_query = f"""
        MATCH (p:Persona)-[*1..2]-(related)
        WHERE toLower(toString(p.id)) CONTAINS '{possible_name}'
        RETURN p, related LIMIT 50
        """
        
        print(f"   🔧 Query Inyectada: {cypher_query.strip()}")
        
        print("   ⚡ Ejecutando en Neo4j...")
        results = graph.query(cypher_query)
        print(f"   🔎 Resultados: {len(results)}")

        if not results:
            return {"response": f"No encontré datos para '{possible_name}'. Intenta verificar el nombre."}

        print("   🗣️ Sintetizando...")
        synthesis_prompt = f"""
        Act as a Financial Auditor.
        USER QUERY: "{request.query}"
        EVIDENCE: {str(results)}
        
        INSTRUCTIONS:
        1. Identify the person (Ana Rojas) and describe her Roles (look for PERTENECE_A, DIRIGE).
        2. LIST specific Expenses/Amounts found (look for nodes with money values like 2200000, 150000).
        3. Explain the context (e.g. Negotiation, Transport).
        4. Answer in Spanish.
        """
        final_response = llm.invoke(synthesis_prompt)
        return {"response": final_response.content}

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"response": f"Error técnico: {str(e)}"}

# --- RUTAS DE UPLOAD (Mismo código de siempre) ---
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