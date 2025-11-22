import os
import requests
from fastapi import FastAPI, Request, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel
from typing import Optional

# --- CORRECCIÓN DE IMPORTS (VERSIÓN ESTABLE 0.2) ---
# Usamos las librerías estándar que sí tienes instaladas
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.llms import Ollama
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

# Tu lógica de ingesta
from graph_agent import run_graph_extraction

app = FastAPI(title="PrismaFinance Specialist Agent")

# --- CONFIGURACIÓN ---
INGESTION_SERVICE = os.getenv("INGESTION_URL", "http://ingestion-service:8000")
INTERNAL_UPLOAD_URL = f"{INGESTION_SERVICE}/upload"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "prismafinance123")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
CHAT_MODEL = "qwen2.5:3b"

# --- PROMPT MAESTRO ---
CYPHER_GENERATION_PROMPT = """
Task: Generate a Neo4j Cypher query.
CRITICAL: Do NOT use SQL syntax. Use only MATCH, RETURN, WHERE.

Instructions:
1. **Flexible Matching:** ALWAYS use `toLower(n.id) CONTAINS 'term'`.
2. **Traversal:** Find connections using variable length paths `-[*1..4]-`.
   Example: `MATCH (n)-[*1..4]-(target) WHERE toLower(n.id) CONTAINS 'juan' RETURN n, target`
3. **Return:** Return the full nodes and relationships found.

Schema:
{schema}

The question is:
{question}
"""

# --- CONSTRUCCIÓN DEL AGENTE ---
def get_financial_agent():
    # 1. Conectar
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
    graph.refresh_schema()
    
    # 2. LLM
    llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)

    # 3. Prompt
    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_PROMPT
    )

    # 4. Cadena (Usando imports clásicos)
    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
        validate_cypher=True,
        cypher_prompt=cypher_prompt
    )

    tools = [
        Tool(
            name="GraphExplorer",
            func=cypher_chain.invoke,
            description="Útil para responder preguntas sobre presupuestos, personas, proyectos y relaciones financieras."
        )
    ]

    # 5. Agente
# ... (código anterior de la función sigue igual) ...

    # Personalidad del Agente (ReAct)
    template = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question (in Spanish)

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    # --- CORRECCIÓN AQUÍ ---
    # 1. Calculamos los nombres manualmente
    tool_names_str = ", ".join([t.name for t in tools])
    
    # 2. Creamos el prompt y le inyectamos (partial) los nombres a la fuerza
    prompt = PromptTemplate.from_template(template).partial(
        tool_names=tool_names_str
    )

    # 3. Creamos el agente
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

# --- ENDPOINTS ---
class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "active", "mode": "stable_v0.2"}

@app.post("/chat")
async def chat(request: QueryRequest):
    print(f"🤖 Agente recibiendo: {request.query}")
    try:
        agent = get_financial_agent()
        response = agent.invoke({"input": request.query})
        return {"response": response.get("output", "Sin respuesta.")}
    except Exception as e:
        print(f"❌ Error Agente: {e}")
        return {"response": f"Error: {str(e)}"}

# --- SUBIDA DE ARCHIVOS ---
def process_graph_in_background(file_path: str):
    print(f"🔄 [Background] Procesando: {file_path}")
    try:
        run_graph_extraction(file_path)
        print(f"✅ [Background] Listo.")
    except Exception as e:
        print(f"❌ [Background] Error: {e}")

@app.api_route("/process", methods=["POST", "PUT"])
@app.api_route("/upload", methods=["POST", "PUT"])
async def proxy_upload(request: Request, background_tasks: BackgroundTasks):
    print(f"📥 Recibiendo archivo...")
    content_type = request.headers.get("content-type", "").lower()
    
    # Intentar leer cabecera personalizada primero
    filename = request.headers.get("x-filename", "doc.bin") 
    file_content = b""
    
    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
            for key, val in form.multi_items():
                if isinstance(val, StarletteUploadFile):
                    filename = val.filename; file_content = await val.read(); break
        except: pass
    
    if not file_content:
        try: file_content = await request.body()
        except: pass

    if not file_content: raise HTTPException(400, "Falta archivo")

    # Backup Local
    backup_dir = "/app/backups"
    os.makedirs(backup_dir, exist_ok=True)
    with open(f"{backup_dir}/{filename}", "wb") as f: f.write(file_content)

    # Reenviar
    try:
        files = {'file': (filename, file_content, content_type)}
        res = requests.post(INTERNAL_UPLOAD_URL, files=files, params=dict(request.query_params))
        if res.status_code == 200:
            background_tasks.add_task(process_graph_in_background, f"{backup_dir}/{filename}")
            return res.json()
        return JSONResponse(res.status_code, res.json())
    except Exception as e:
        raise HTTPException(500, str(e))