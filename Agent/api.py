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
CHAT_MODEL = os.getenv("CHAT_MODEL", "ibm/granite-4-h-tiny") 

INGESTION_SERVICE = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
INTERNAL_UPLOAD_URL = f"{INGESTION_SERVICE}/upload"

# --- 3. GESTIÓN DE HISTORIAL (MEMORIA) ---
# Crea un archivo 'chat_history.db' local para guardar las conversaciones
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

# --- 4. PROMPTS ---
# El prompt mejorado para encontrar entidades (Fase 1)
EXTRACTION_PROMPT = """
You are an Expert Search Term Extractor for a Financial Knowledge Graph.
Your goal is to identify the single most important "anchor" term from the user's query.

DATABASE SCHEMA:
- Personas (e.g., 'Pedro Maza', 'Ana Rojas')
- Organizaciones (e.g., 'Metso', 'Candelaria')
- Proyectos (e.g., 'Mantenimiento Planta')
- Conceptos (e.g., 'Bono', 'Presupuesto')

RULES:
1. Identify the MAIN SUBJECT. It can be a Person, Organization, Project, or Concept.
2. Remove stop words ("el", "la", "de") and verbs ("reporto", "gastó").
3. DO NOT output labels. Just the raw term.
4. IF multiple entities exist, prioritize the most specific proper name.
5. IF the query is general (e.g., "resumen del mes"), output "GENERAL".

Example: "Gastos de Ana Rojas" -> Ana Rojas
"""

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"  # ID para recordar la conversación

@app.get("/health")
def health():
    return {"status": "active", "mode": "final_audit_v9_memory"}

@app.post("/chat")
async def chat(request: QueryRequest):
    print(f"🔥 [CHAT] Sesión: {request.session_id} | Pregunta: {request.query}")
    try:
        # Conexiones
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_base=OPENAI_BASE,
            openai_api_key=OPENAI_KEY,
            temperature=0
        )
        
        # --- FASE 1: EXTRACCIÓN DE ENTIDAD ---
        print("   🧠 Extrayendo entidad...")
        extract_msgs = [SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=request.query)]
        entity_name_raw = llm.invoke(extract_msgs).content
        entity_name = entity_name_raw.replace("\n", "").replace('"', '').strip()
        print(f"   🎯 Entidad Clave: '{entity_name}'")
        
        # --- FASE 2: BÚSQUEDA EN NEO4J ---
        context_str = ""
        if "GENERAL" in entity_name.upper():
            # Búsqueda amplia si no hay entidad clara
            cypher_query = "MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50"
        else:
            # Búsqueda dirigida
            cypher_query = f"""
            MATCH (p)-[*1..2]-(related)
            WHERE toLower(toString(p.id)) CONTAINS toLower('{entity_name}')
            RETURN p, related LIMIT 100
            """
        
        try:
            results = graph.query(cypher_query)
            context_str = str(results) if results else "No se encontraron registros exactos en el grafo."
            print(f"   🔎 Datos encontrados: {len(results)} registros")
        except Exception as e:
            context_str = f"Error consultando grafo: {str(e)}"

        # --- FASE 3: GENERACIÓN CON MEMORIA ---
        print("   🗣️ Generando respuesta...")
        
        # Prompt que incluye el Historial ({chat_history})
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un asistente financiero experto. Responde en Español."),
            ("system", "Tus respuestas deben basarse en el CONTEXTO DEL GRAFO proporcionado."),
            ("system", "CONTEXTO DEL GRAFO:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"), # Aquí entra la memoria
            ("human", "{input}")
        ])

        chain = qa_prompt | llm

        # Cadena con manejo automático de historial
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        response = chain_with_history.invoke(
            {"input": request.query, "context": context_str},
            config={"configurable": {"session_id": request.session_id}}
        )

        return {"response": response.content}

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {"response": f"Error técnico: {str(e)}"}

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