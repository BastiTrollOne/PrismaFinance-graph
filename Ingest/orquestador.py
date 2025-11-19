# orquestador.py
import os
import httpx
from fastapi import FastAPI, UploadFile
from langchain_community.graphs import Neo4jGraph
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # (Ajustado para Ollama)
from langchain_community.chat_models import ChatOllama

# --- CONFIGURACIÓN ---
app = FastAPI()

# 1. Conexión al Grafo (usa las vars del .env del docker-compose)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# 2. Conexión al LLM Local
llm = ChatOllama(model="llama3:8b") # O el modelo que uses

# 3. Dirección de tu MCP de Ingesta
INGEST_MCP_URL = "http://localhost:8000/process" # (O el nombre del servicio en Docker)

# --- MODELO DE DATOS (Para extracción de entidades) ---
class ExtractedEntities(BaseModel):
    """Información extraída del chunk de texto."""
    personas: list[str] = Field(description="Nombres de personas")
    kpis: list[str] = Field(description="Indicadores Clave de Desempeño (ej. 'reducción del 40%')")
    proyectos: list[str] = Field(description="Nombres de proyectos o minas (ej. 'Candelaria')")

# --- LÓGICA DE EXTRACCIÓN (Paso 3) ---
async def extract_entities_from_chunk(text_chunk: str) -> ExtractedEntities:
    """Usa el LLM para extraer entidades estructuradas de un chunk."""
    
    # Crea un LLM "estructurado" que debe responder con el formato Pydantic
    structured_llm = llm.with_structured_output(ExtractedEntities)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto en finanzas mineras. Extrae las entidades solicitadas del texto. Si una entidad no está presente, devuelve una lista vacía."),
        ("human", "{texto}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        entities = await chain.ainvoke({"texto": text_chunk})
        return entities
    except Exception as e:
        print(f"Error en extracción LLM: {e}")
        return ExtractedEntities(personas=[], kpis=[], proyectos=[])

# --- LÓGICA DEL GRAFO (Paso 4) ---
async def update_graph_with_entities(doc_id: str, chunk_id: int, entities: ExtractedEntities):
    """Escribe los nodos y relaciones en Neo4j usando Cypher."""
    
    # 1. Crear el nodo del Chunk (vinculado al Documento)
    graph.query(
        """
        MERGE (d:Document {id: $doc_id})
        MERGE (c:Chunk {id: $chunk_id, doc_id: $doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        params={"doc_id": doc_id, "chunk_id": f"{doc_id}_chunk_{chunk_id}"}
    )
    
    # 2. Conectar Personas
    for persona in entities.personas:
        graph.query(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (p:Persona {name: $name})
            MERGE (c)-[:MENTIONS_PERSON]->(p)
            """,
            params={"chunk_id": f"{doc_id}_chunk_{chunk_id}", "name": persona}
        )

    # 3. Conectar KPIs
    for kpi in entities.kpis:
        graph.query(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (k:KPI {name: $name})
            MERGE (c)-[:MENTIONS_KPI]->(k)
            """,
            params={"chunk_id": f"{doc_id}_chunk_{chunk_id}", "name": kpi}
        )
    # ... (repetir para Proyectos) ...


# --- ENDPOINT PRINCIPAL (EL ORQUESTADOR) ---
@app.post("/ingest_document/")
async def ingest_document(file: UploadFile):
    
    file_content = await file.read()
    
    # --- Paso 2: Llama a tu MCP de Ingesta ---
    files = {'file': (file.filename, file_content, file.content_type)}
    params = {'emit_multi': 'true'} # Pedir chunks separados
    
    async with httpx.AsyncClient() as client:
        response = await client.post(INGEST_MCP_URL, files=files, params=params)
    
    if response.status_code != 200:
        return {"error": "Fallo el MCP de Ingesta", "detail": response.text}

    ingest_data = response.json()
    chunks = ingest_data.get("documents", [])
    doc_id = chunks[0]["metadata"].get("doc_id") # Asumimos que todos comparten doc_id

    # --- Pasos 3, 4 y 5 ---
    for i, chunk in enumerate(chunks):
        chunk_text = chunk["page_content"]
        
        # Paso 3: Extracción de Entidades
        entities = await extract_entities_from_chunk(chunk_text)
        
        # Paso 4: Actualización del Grafo
        await update_graph_with_entities(doc_id, i, entities)
        
        # Paso 5: (Aquí iría la llamada al Vector Store)
        # vector_store.add_text(chunk_text, metadata=chunk["metadata"])

    return {"status": "success", "doc_id": doc_id, "chunks_processed": len(chunks)}