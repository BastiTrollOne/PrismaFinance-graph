import os
import requests
import re
import time
import traceback
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel

# --- IMPORTS DE LANGCHAIN & IA ---
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from graph_agent import run_graph_extraction

app = FastAPI(title="PrismaFinance Nuclear Agent")

# --- 1. HABILITAR CORS (ACCESO EXTERNO) ---
# Esto permite que tu chat funcione desde cualquier página web o IP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes (ajustar en prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CONFIGURACIÓN ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "prismafinance123")

# Ajusta la IP de tu servidor de modelos aquí si cambió
OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://192.168.50.1:8900/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-needed")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3-4b-2507") 

INGESTION_SERVICE = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
INTERNAL_UPLOAD_URL = f"{INGESTION_SERVICE}/upload"

# --- 3. GESTIÓN DE HISTORIAL (MEMORIA) ---
# Crea un archivo 'chat_history.db' local para guardar las conversaciones
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

# --- 4. PROMPTS ---
# El prompt mejorado para encontrar entidades (Fase 1)
# --- En Agent/api.py ---

EXTRACTION_PROMPT = """
You are the **Intent Classifier & Entity Extractor** for a Financial Knowledge Graph.
Your goal is to translate user questions into precise search signals.

### 🚨 CRITICAL RULES (Read Carefully):

1. **NAMED ENTITY RECOGNITION (High Priority)**
   - If the user asks about a Person, Organization, Project, or Role, extract the **Exact Name**.
   - **FIX SPACING**: If the user types "Javierasilva", output "Javiera Silva".
   - **IGNORE FILLER**: Ignore phrases like "in the documents", "in the graph", "loaded files", "tell me about".
   - *Example:* "Que hace Javiera Silva en los archivos" -> Javiera Silva
   - *Example:* "Cargo de CarlosVidela" -> Carlos Videla

2. **NUMERIC & FINANCIAL SEARCH**
   - If the user asks for a specific amount, extract **ONLY** the number (digits).
   - *Example:* "Monto de 60000" -> 60000
   - *Example:* "Facturas mayores a 1000" -> 1000

3. **GLOBAL / DISCOVERY INTENT (General)**
   - If the user asks for a **Summary**, **Overview**, **Relationships** (without a specific name), **Patterns**, or "What is in the documents?", output exactly: **GENERAL**.
   - *Example:* "Relaciona los documentos cargados" -> GENERAL
   - *Example:* "Dame un resumen ejecutivo" -> GENERAL
   - *Example:* "¿De qué trata el excel cargado?" -> GENERAL

4. **CONCEPTUAL SEARCH**
   - If the user asks about a concept (e.g., "Interest Rate", "Budget", "Debt"), extract the concept in its simplest form.
   - *Example:* "Cual es la tasa de interés" -> Tasa de interés

### 🛡️ SAFETY & CLEANING:
- **Output Format:** JUST the raw term or "GENERAL". No JSON, no labels, no quotes.
- If multiple entities exist, prioritize the **Proper Name**.
- If the query is "Hola" or trivial conversation, output: **GENERAL**.

### TEST CASES:
User: "Gastos de Ana" -> Ana
User: "Monto 500.000" -> 500000
User: "Resumen de documentos" -> GENERAL
User: "Que hace joseperez" -> Jose Perez
"""

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"  # ID para recordar la conversación

@app.get("/health")
def health():
    return {"status": "active", "mode": "final_audit_v9_memory"}

@app.post("/chat")
async def chat(request: QueryRequest):
    # 1. FILTRO ANTI-RUIDO
    if request.query.strip().startswith("### Task") or "Generate a concise" in request.query:
        return {"response": "System Task Processed"} 

    print(f"🔥 [CHAT] Sesión: {request.session_id} | Pregunta: {request.query}")
    
    try:
        # 2. CONEXIONES
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_base=OPENAI_BASE,
            openai_api_key=OPENAI_KEY,
            temperature=0
        )
        
        # 3. INTENCIÓN
        print("   🧠 Analizando intención...")
        extract_msgs = [SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=request.query)]
        entity_name = llm.invoke(extract_msgs).content.replace("\n", "").replace('"', '').replace("'", "").strip()
        print(f"   🎯 Intención: '{entity_name}'")
        
        # 4. ESTRATEGIA OPTIMIZADA (LIGHTWEIGHT)
        context_str = ""
        
        if "GENERAL" in entity_name.upper() or entity_name == "":
            print("   🌍 Búsqueda GLOBAL (Optimizada)...")
            # TRUCO: No traemos el objeto entero (n), solo sus propiedades clave y un recorte del texto.
            # Esto evita traer 10MB de datos si los nodos tienen PDFs enteros dentro.
            cypher_query = """
            MATCH (n)-[r]->(m)
            WITH n, count(r) as rel_count, collect(r) as rels, collect(m) as targets
            ORDER BY rel_count DESC
            LIMIT 5
            UNWIND rels as r
            UNWIND targets as m
            RETURN n.id as Source, type(r) as Rel, m.id as Target, left(toString(m.text), 150) as ContextSnippet
            LIMIT 30
            """
        else:
            print(f"   🔍 Búsqueda ESPECÍFICA: '{entity_name}'")
            cypher_query = f"""
            MATCH (p)-[r]-(related)
            WHERE toLower(toString(p.id)) CONTAINS toLower('{entity_name}')
            RETURN p.id as Source, type(r) as Rel, related.id as Target, left(toString(related.text), 150) as ContextSnippet
            LIMIT 30
            """
        
        # 5. EJECUCIÓN
        try:
            # graph.query devuelve una lista de dicts, mucho más ligera ahora
            results = graph.query(cypher_query)
            
            # Convertimos resultados a texto compacto
            # (Source: X, Rel: Y, Target: Z, Snippet: ...)
            context_list = []
            for row in results:
                context_list.append(str(row))
            full_context = "\n".join(context_list)

            # Recorte estricto para velocidad (7000 chars ~= 1500 tokens)
            if len(full_context) > 7000:
                context_str = full_context[:7000] + "... [truncado]"
            else:
                context_str = full_context

            print(f"   🔎 Datos: {len(results)} filas. Peso contexto: {len(context_str)} chars.")
            
        except Exception as e:
            print(f"   ❌ Error Cypher: {e}")
            context_str = f"Error: {str(e)}"

        # 6. GENERACIÓN
        print("   🗣️ Generando respuesta...")
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un analista experto. Responde en Español."),
            ("system", "Básate SOLO en este resumen de relaciones del grafo:\n{context}"),
            ("system", "Si el contexto es limitado, explica las relaciones clave que ves."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        chain = qa_prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            chain, get_session_history,
            input_messages_key="input", history_messages_key="chat_history"
        )

        response = chain_with_history.invoke(
            {"input": request.query, "context": context_str},
            config={"configurable": {"session_id": request.session_id}}
        )

        return {"response": response.content}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"response": "Error técnico. Ver logs."}

# --- RUTAS DE UPLOAD (Sin cambios) ---
@app.api_route("/process", methods=["POST", "PUT"])
@app.api_route("/upload", methods=["POST", "PUT"])
async def proxy_upload(request: Request, background_tasks: BackgroundTasks):
    print(f"📥 [UPLOAD] Solicitud entrante...")
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
    clean_name = re.sub(r'[^\w\-_\.]', '_', os.path.splitext(os.path.basename(filename))[0])[:50]
    final_filename = f"{int(time.time())}_{clean_name}{os.path.splitext(filename)[1]}"
    local_path = f"{backup_dir}/{final_filename}"
    
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